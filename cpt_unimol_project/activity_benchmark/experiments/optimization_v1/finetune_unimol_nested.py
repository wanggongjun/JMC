"""Uni-Mol v1 supervised fine-tuning with scaffold-held-out evaluation.

One outer scaffold fold is evaluated at a time. Within the remaining outer
training data, a separate scaffold validation partition drives early stopping.
Only the inner train+validation rows are passed to DataHub during fitting; the
outer test rows are encoded separately after the checkpoint is fixed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
EXP_BASE = Path(__file__).resolve().parent
DATA = ROOT / "data/processed/modeling_dataset.csv"
CACHE = ROOT / "data/processed/conformer_features.pkl"
WEIGHTS = ROOT / "models/unimol/mol_pre_all_h_220816.pt"
DICT = ROOT / "models/unimol/mol.dict.txt"
TASKS = ("hepg2_pIC50", "hct116_pIC50")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def enable_mps(Trainer, torch):
    original_init = Trainer.__init__

    def mps_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.device = torch.device("mps")
        self.ddp = False

    def mps_decorate(self, batch):
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input = {k: v.to(self.device) for k, v in net_input.items()}
        else:
            net_input = {"net_input": net_input.to(self.device)}
        return net_input, net_target.to(self.device).float()

    Trainer.__init__ = mps_init
    Trainer.decorate_torch_batch = mps_decorate


def make_frame(subset: pd.DataFrame, by_smiles: dict, label_col: str | None = None) -> pd.DataFrame:
    rows = []
    for row in subset.itertuples(index=False):
        s = str(row.smiles)
        conf = by_smiles[s]
        if conf.get("atoms") is None or conf.get("coords") is None:
            raise RuntimeError(f"Missing cached conformer: {s}")
        item = {"SMILES": s, "atoms": conf["atoms"],
                "coordinates": np.asarray(conf["coords"], dtype=np.float32)}
        if label_col is not None:
            item["TARGET"] = float(getattr(row, label_col))
        rows.append(item)
    return pd.DataFrame(rows)


def main():
    import torch
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from unimol_tools.data import DataHub
    from unimol_tools.models.nnmodel import NNModel, NNDataset
    from unimol_tools.tasks import Trainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--max-outer-folds", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke test only")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--output-dir", default="finetune_nested_cv")
    args = parser.parse_args()
    global EXP
    EXP = EXP_BASE / args.output_dir
    EXP.mkdir(parents=True, exist_ok=True)
    logdir = EXP / "fold_logs"
    logdir.mkdir(exist_ok=True)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if device == "mps":
        enable_mps(Trainer, torch)
    torch.set_num_threads(max(1, min(4, int(os.environ.get("OMP_NUM_THREADS", "4")))))

    data = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    if args.max_rows:
        data = data.head(args.max_rows).copy()
    with CACHE.open("rb") as f:
        conformers = pickle.load(f)
    by_smiles = {str(row["smiles"]): row for row in conformers}
    data = data.loc[data.smiles.astype(str).isin(by_smiles)].copy().reset_index(drop=True)
    scaffold_by_smiles = {}
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    for s in data.smiles.astype(str):
        mol = Chem.MolFromSmiles(s)
        scaffold_by_smiles[s] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    params = {
        "task": "regression", "data_type": "molecule", "model_name": "unimolv1",
        "pretrained_model_path": str(WEIGHTS), "pretrained_dict_path": str(DICT),
        "remove_hs": False, "max_atoms": 320,
        "target_cols": "TARGET", "target_col_prefix": "TARGET", "smiles_col": "SMILES",
        "target_normalize": "none", "anomaly_clean": False, "smi_strict": False,
        "split": "random", "split_group_col": "scaffold", "kfold": 1,
        "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "epochs": args.epochs, "patience": 6, "metrics": "mse",
        "warmup_ratio": 0.06, "max_norm": 1.0,
        "use_cuda": False, "use_amp": False, "use_ddp": False,
        "conf_cache_level": 0, "seed": 20260925,
    }
    cfg = {"device": device, "parameters": params,
           "unimol_weight_sha256": sha256(WEIGHTS), "dictionary_sha256": sha256(DICT),
           "modeling_dataset_sha256": sha256(DATA), "conformer_cache_sha256": sha256(CACHE),
           "protocol": "outer GroupKFold(scaffold); inner GroupShuffleSplit 12% scaffold validation; early stopping by validation MSE",
           "outer_folds": args.outer_folds, "inner_validation_fraction": 0.12,
           "target_scaling": "standardized using inner-training mean/std only; predictions are returned to pIC50 scale"}
    (EXP / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    metric_path = EXP / "outer_fold_metrics.csv"
    pred_path = EXP / "outer_oof_predictions.csv"
    split_path = EXP / "outer_scaffold_folds.csv"
    completed = set()
    if metric_path.exists():
        previous = pd.read_csv(metric_path)
        completed = {(str(r.task), int(r.outer_fold)) for r in previous.itertuples()}
    metric_rows, pred_rows, split_rows = [], [], []
    if metric_path.exists():
        metric_rows = pd.read_csv(metric_path).to_dict(orient="records")
    if pred_path.exists():
        pred_rows = pd.read_csv(pred_path).to_dict(orient="records")
    split_rows = pd.read_csv(split_path).to_dict(orient="records") if split_path.exists() else []

    for task in args.tasks:
        observed = data.loc[data[task].notna(), ["smiles", task]].reset_index(drop=True)
        y = observed[task].to_numpy(dtype=np.float32)
        groups = np.asarray([scaffold_by_smiles[s] for s in observed.smiles.astype(str)])
        outer = GroupKFold(n_splits=args.outer_folds)
        folds = list(outer.split(np.arange(len(observed)), y, groups))
        if args.max_outer_folds:
            folds = folds[:args.max_outer_folds]
        for fold, (outer_train, outer_test) in enumerate(folds):
            if (task, fold) in completed:
                continue
            fold_started = time.time()
            inner_splitter = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=20260925 + fold)
            fit_rel, val_rel = next(inner_splitter.split(outer_train, y[outer_train], groups[outer_train]))
            inner_train = outer_train[fit_rel]
            inner_val = outer_train[val_rel]
            if (set(groups[inner_train]) & set(groups[inner_val])
                    or set(groups[outer_train]) & set(groups[outer_test])):
                raise RuntimeError("Scaffold overlap in train/validation/test partition")
            roles = {int(i): "inner_train" for i in inner_train}
            roles.update({int(i): "early_stop_validation" for i in inner_val})
            roles.update({int(i): "outer_test" for i in outer_test})
            for i, role in roles.items():
                split_rows.append({"task": task, "outer_fold": fold,
                                   "smiles": str(observed.iloc[i].smiles),
                                   "scaffold": groups[i], "role": role})
            inner_ids = np.concatenate([inner_train, inner_val])
            target_mean = float(np.mean(y[inner_train]))
            target_std = float(np.std(y[inner_train], ddof=0))
            if not np.isfinite(target_std) or target_std <= 1e-8:
                raise RuntimeError(f"Invalid training target scale for {task}/fold{fold}")
            trainval = observed.iloc[inner_ids].reset_index(drop=True)
            frame = make_frame(trainval, by_smiles, task)
            frame["TARGET"] = (frame["TARGET"] - target_mean) / target_std
            out_dir = EXP / "checkpoints" / task / f"fold{fold}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)
            log_file = logdir / f"{task}_fold{fold}.log"
            (out_dir / "fold_config.json").write_text(json.dumps({
                "task": task, "outer_fold": fold, "n_fit": len(inner_train),
                "n_validation": len(inner_val), "n_outer_test": len(outer_test),
                "train_scaffolds": len(set(groups[inner_train])),
                "validation_scaffolds": len(set(groups[inner_val])),
                "test_scaffolds": len(set(groups[outer_test])),
                "target_mean_fit_only": target_mean, "target_std_fit_only": target_std,
                "outer_scaffold_overlap": 0,
                "inner_scaffold_overlap": 0,
            }, indent=2) + "\n", encoding="utf-8")

            print(f"START {task} outer={fold+1}/{args.outer_folds} device={device} fit={len(inner_train)} val={len(inner_val)} test={len(outer_test)}", flush=True)
            datahub = DataHub(data=frame, is_train=True, save_path=str(out_dir),
                              **params)
            train_local = np.arange(len(inner_train), dtype=int)
            val_local = np.arange(len(inner_train), len(inner_train) + len(inner_val), dtype=int)
            datahub.data["split_nfolds"] = [(train_local, val_local)]
            datahub.data["kfold"] = 1
            trainer = Trainer(save_path=str(out_dir), **params)
            model = NNModel(datahub.data, trainer, **params)
            model.run()
            val_pred = np.asarray(model.cv["pred"]).reshape(-1)
            val_true = np.asarray(datahub.data["target"]).reshape(-1)
            val_rmse = float(np.sqrt(mean_squared_error(val_true[val_local], val_pred[val_local])) * target_std)
            del model, trainer, datahub
            gc.collect()

            test_subset = observed.iloc[outer_test].reset_index(drop=True)
            test_frame = make_frame(test_subset, by_smiles, task)
            test_frame["TARGET"] = (test_frame["TARGET"] - target_mean) / target_std
            testhub = DataHub(data=test_frame, is_train=False, save_path=str(out_dir), **params)
            test_trainer = Trainer(save_path=str(out_dir), **params)
            test_model = NNModel(testhub.data, test_trainer, **params)
            test_model.evaluate(test_trainer, str(out_dir))
            pred_z = np.asarray(test_model.cv["test_pred"]).reshape(-1)
            pred = pred_z * target_std + target_mean
            truth = test_subset[task].to_numpy(dtype=np.float32)
            if len(pred) != len(truth) or not np.isfinite(pred).all():
                raise RuntimeError(f"Invalid outer predictions for {task}/fold{fold}")
            rmse = float(np.sqrt(mean_squared_error(truth, pred)))
            mae = float(mean_absolute_error(truth, pred))
            r2 = float(r2_score(truth, pred))
            metric_rows.append({"task": task, "outer_fold": fold, "model": "unimol_v1_finetuned",
                                "validation_rmse": val_rmse, "n_test": len(truth),
                                "rmse": rmse, "mae": mae, "r2": r2,
                                "pearson": float(np.corrcoef(truth, pred)[0, 1]) if np.std(pred) > 0 else 0.0,
                                "spearman": float(pd.Series(truth).corr(pd.Series(pred), method="spearman")),
                                "n_fit": len(inner_train), "n_validation": len(inner_val),
                                "device": device, "epochs_max": args.epochs,
                                "learning_rate": args.learning_rate, "batch_size": args.batch_size})
            for i, row_id in enumerate(outer_test):
                s = str(observed.iloc[row_id].smiles)
                pred_rows.append({"task": task, "outer_fold": fold, "smiles": s,
                                  "scaffold": groups[row_id], "y_true": float(truth[i]),
                                  "model": "unimol_v1_finetuned", "y_pred": float(pred[i])})
            pd.DataFrame(metric_rows).to_csv(metric_path, index=False)
            pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
            pd.DataFrame(split_rows).drop_duplicates(
                subset=["task", "outer_fold", "smiles"], keep="last"
            ).to_csv(split_path, index=False)
            if not args.keep_checkpoints:
                shutil.rmtree(out_dir)
            del testhub, test_trainer, test_model
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            completed.add((task, fold))
            print(f"DONE {task} outer={fold+1}/{args.outer_folds} val_rmse={val_rmse:.4f} test_rmse={rmse:.4f} elapsed={time.time()-fold_started:.1f}s", flush=True)

    metrics_df = pd.DataFrame(metric_rows)
    summary = []
    for task, part in metrics_df.groupby("task"):
        summary.append({"task": task, "model": "unimol_v1_finetuned", "n_outer_folds": len(part),
                        **{f"{key}_mean": float(part[key].mean()) for key in ("rmse", "mae", "r2", "pearson", "spearman")},
                        **{f"{key}_std": float(part[key].std(ddof=1)) for key in ("rmse", "mae", "r2", "pearson", "spearman")}})
    pd.DataFrame(summary).to_csv(EXP / "pooled_nested_cv_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    # Imports kept below the module docstring so torch's MPS fallback can be set first.
    import numpy as np
    import pandas as pd
    main()
