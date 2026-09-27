"""Uni-Mol2 84M supervised fine-tuning with scaffold-held-out evaluation.

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
ROOT = Path(__file__).resolve().parents[2]
EXP_BASE = Path(__file__).resolve().parent
RUNTIME = EXP_BASE / "unimol2_runtime"
os.environ["UNIMOL_WEIGHT_DIR"] = str(ROOT / "models/unimol2")
import sys
sys.path.insert(0, str(RUNTIME))

import numpy as np
import pandas as pd
DATA = ROOT / "data/processed/modeling_dataset.csv"
CACHE = ROOT / "data/processed/conformer_features.pkl"
WEIGHTS = ROOT / "models/unimol2/modelzoo/84M/checkpoint.pt"
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
                "coordinates": [[float(v) for v in coord] for coord in conf["coords"]]}
        if label_col is not None:
            item["TARGET"] = float(getattr(row, label_col))
        rows.append(item)
    return pd.DataFrame(rows)


def runtime_data(frame: pd.DataFrame) -> dict:
    """The isolated Uni-Mol2 runtime accepts dict/CSV input, not a DataFrame."""
    return {str(column): frame[column].tolist() for column in frame.columns}


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
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-atoms", type=int, default=160,
                        help="Padding limit; set at or above the largest observed atom count")
    parser.add_argument("--learning-rate", type=float, default=0.0,
                        help="Override the preset learning rate for the selected freeze mode")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--max-outer-folds", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke test only")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--freeze-mode", choices=("head", "last2", "full"), default="last2")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--folds-file", default="",
                        help="Optional frozen task/smiles/outer_fold manifest")
    parser.add_argument("--output-dir", default="finetune_unimol2_nested_cv")
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
    observed_max_atoms = max(len(by_smiles[str(s)]["atoms"]) for s in data.smiles.astype(str))
    if observed_max_atoms > args.max_atoms:
        raise RuntimeError(
            f"--max-atoms={args.max_atoms} would truncate a molecule with "
            f"{observed_max_atoms} atoms; raise the limit before training"
        )
    scaffold_by_smiles = {}
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    for s in data.smiles.astype(str):
        mol = Chem.MolFromSmiles(s)
        scaffold_by_smiles[s] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    learning_rate = args.learning_rate or {"head": 1e-3, "last2": 1e-5, "full": 5e-6}[args.freeze_mode]
    params = {
        "task": "regression", "data_type": "molecule", "model_name": "unimolv2",
        "model_size": "84m",
        "remove_hs": False, "max_atoms": args.max_atoms,
        "target_cols": "TARGET", "target_col_prefix": "TARGET", "smiles_col": "SMILES",
        "target_normalize": "none", "anomaly_clean": False, "smi_strict": False,
        "split": "random", "split_group_col": "scaffold", "kfold": 1,
        "learning_rate": learning_rate, "batch_size": args.batch_size,
        "epochs": args.epochs, "patience": args.patience, "metrics": "mse",
        "warmup_ratio": 0.06, "max_norm": 1.0,
        "use_cuda": False, "use_amp": False, "use_ddp": False,
        "conf_cache_level": 0, "seed": args.seed,
    }
    if args.freeze_mode == "head":
        params["freeze_layers"] = ["classification_head"]
        params["freeze_layers_reversed"] = True
    elif args.freeze_mode == "last2":
        params["freeze_layers"] = ["encoder.layers.10", "encoder.layers.11", "classification_head"]
        params["freeze_layers_reversed"] = True
    else:
        params["freeze_layers"] = None
        params["freeze_layers_reversed"] = False
    cfg = {"device": device, "parameters": params,
           "unimol_weight_sha256": sha256(WEIGHTS),
           "modeling_dataset_sha256": sha256(DATA), "conformer_cache_sha256": sha256(CACHE),
           "model_name": "unimolv2", "model_size": "84m", "freeze_mode": args.freeze_mode,
           "max_atoms_padding_limit": args.max_atoms,
           "observed_max_atoms": observed_max_atoms,
           "smoke_only": bool(args.max_rows or args.max_outer_folds),
           "protocol": "outer scaffold holdout; inner GroupShuffleSplit 12% scaffold validation; early stopping by validation MSE",
           "outer_folds": args.outer_folds, "inner_validation_fraction": 0.12,
           "outer_fold_manifest": args.folds_file or "deterministic GroupKFold",
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
        if args.folds_file:
            folds_path = Path(args.folds_file)
            if not folds_path.is_absolute():
                folds_path = ROOT / folds_path
            fold_table = pd.read_csv(folds_path)
            task_folds = fold_table.loc[fold_table.task == task, ["smiles", "outer_fold"]]
            fold_map = dict(zip(task_folds.smiles.astype(str), task_folds.outer_fold.astype(int)))
            if set(fold_map) != set(observed.smiles.astype(str)):
                raise RuntimeError(f"Frozen fold manifest coverage mismatch for {task}")
            row_folds = observed.smiles.astype(str).map(fold_map).to_numpy(dtype=int)
            folds = [(np.flatnonzero(row_folds != fold), np.flatnonzero(row_folds == fold))
                     for fold in range(args.outer_folds)]
            if any(len(test) == 0 for _, test in folds):
                raise RuntimeError(f"Empty outer fold in frozen manifest for {task}")
        else:
            outer = GroupKFold(n_splits=args.outer_folds)
            folds = list(outer.split(np.arange(len(observed)), y, groups))
        if args.max_outer_folds:
            folds = folds[:args.max_outer_folds]
        for fold, (outer_train, outer_test) in enumerate(folds):
            if (task, fold) in completed:
                continue
            fold_started = time.time()
            inner_splitter = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=args.seed + fold)
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
            datahub = DataHub(data=runtime_data(frame), is_train=True, save_path=str(out_dir),
                              **params)
            train_local = np.arange(len(inner_train), dtype=int)
            val_local = np.arange(len(inner_train), len(inner_train) + len(inner_val), dtype=int)
            datahub.data["split_nfolds"] = [(train_local, val_local)]
            datahub.data["kfold"] = 1
            trainer = Trainer(save_path=str(out_dir), **params)
            if str(trainer.device) != str(device):
                raise RuntimeError(f"Trainer device mismatch: expected {device}, got {trainer.device}")
            print(f"TRAIN_DEVICE={trainer.device} trainable_mode={args.freeze_mode}", flush=True)
            model = NNModel(datahub.data, trainer, **params)
            trainable_count = sum(parameter.numel() for parameter in model.model.parameters()
                                  if parameter.requires_grad)
            print(f"TRAINABLE_PARAMETERS={trainable_count}", flush=True)
            model.run()
            val_pred = np.asarray(model.cv["pred"]).reshape(-1)
            val_true = np.asarray(datahub.data["target"]).reshape(-1)
            val_rmse = float(np.sqrt(mean_squared_error(val_true[val_local], val_pred[val_local])) * target_std)
            del model, trainer, datahub
            gc.collect()

            test_subset = observed.iloc[outer_test].reset_index(drop=True)
            test_frame = make_frame(test_subset, by_smiles, task)
            test_frame["TARGET"] = (test_frame["TARGET"] - target_mean) / target_std
            testhub = DataHub(data=runtime_data(test_frame), is_train=False,
                              save_path=str(out_dir), **params)
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
            metric_rows.append({"task": task, "outer_fold": fold, "model": f"unimol_v2_finetuned_{args.freeze_mode}",
                                "validation_rmse": val_rmse, "n_test": len(truth),
                                "rmse": rmse, "mae": mae, "r2": r2,
                                "pearson": float(np.corrcoef(truth, pred)[0, 1]) if np.std(pred) > 0 else 0.0,
                                "spearman": float(pd.Series(truth).corr(pd.Series(pred), method="spearman")),
                                "n_fit": len(inner_train), "n_validation": len(inner_val),
                                "device": device, "epochs_max": args.epochs,
                                "learning_rate": learning_rate, "batch_size": args.batch_size,
                                "freeze_mode": args.freeze_mode,
                                "patience": args.patience})
            for i, row_id in enumerate(outer_test):
                s = str(observed.iloc[row_id].smiles)
                pred_rows.append({"task": task, "outer_fold": fold, "smiles": s,
                                  "scaffold": groups[row_id], "y_true": float(truth[i]),
                                  "model": f"unimol_v2_finetuned_{args.freeze_mode}", "y_pred": float(pred[i])})
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
        summary.append({"task": task, "model": f"unimol_v2_finetuned_{args.freeze_mode}", "n_outer_folds": len(part),
                        **{f"{key}_mean": float(part[key].mean()) for key in ("rmse", "mae", "r2", "pearson", "spearman")},
                        **{f"{key}_std": float(part[key].std(ddof=1)) for key in ("rmse", "mae", "r2", "pearson", "spearman")}})
    pd.DataFrame(summary).to_csv(EXP / "pooled_nested_cv_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    # Imports kept below the module docstring so torch's MPS fallback can be set first.
    import numpy as np
    import pandas as pd
    main()
