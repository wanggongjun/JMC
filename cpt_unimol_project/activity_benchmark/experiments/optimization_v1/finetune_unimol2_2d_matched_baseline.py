#!/usr/bin/env python3
"""Nested 2D comparator using the exact inner train/validation/outer test rows as Uni-Mol2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import assay_context_3d_benchmark as base

ROOT = base.ROOT
DATA = ROOT / "data/processed/modeling_dataset.csv"
FEATURES = ROOT / "results/2d_morgan_rdkit_features.npz"
TASKS = ("hepg2_pIC50", "hct116_pIC50")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def get_folds(observed: pd.DataFrame, groups: np.ndarray, task: str,
              n_folds: int, folds_file: str):
    if not folds_file:
        return list(GroupKFold(n_splits=n_folds).split(
            np.arange(len(observed)), observed.y.to_numpy(), groups))
    path = Path(folds_file)
    if not path.is_absolute():
        path = ROOT / path
    table = pd.read_csv(path)
    part = table.loc[table.task == task, ["smiles", "outer_fold"]]
    mapping = dict(zip(part.smiles.astype(str), part.outer_fold.astype(int)))
    if set(mapping) != set(observed.smiles.astype(str)):
        raise RuntimeError(f"Frozen fold manifest coverage mismatch: {task}")
    fold_ids = observed.smiles.astype(str).map(mapping).to_numpy(dtype=int)
    splits = [(np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f))
              for f in range(n_folds)]
    if any(not len(test) for _, test in splits):
        raise RuntimeError(f"Empty outer fold: {task}")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-file", default="experiments/optimization_v1/nested_cv/outer_scaffold_folds.csv")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--output-dir", default="experiments/optimization_v1/finetune_unimol2_2d_matched")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    f2 = np.load(FEATURES, allow_pickle=False)
    feature_map = {str(s): i for i, s in enumerate(f2["smiles"])}
    if set(data.smiles.astype(str)) - set(feature_map):
        raise RuntimeError("2D feature coverage is incomplete")
    x2 = f2["features"].astype(np.float32)
    if x2.shape[1] < 2048 or not np.isfinite(x2).all():
        raise RuntimeError("Invalid 2D feature matrix")
    molecule_ids = data.smiles.astype(str).map(feature_map).to_numpy(dtype=int)
    data["scaffold"] = data.smiles.astype(str).map(scaffold)

    scores, selections, predictions, full_predictions, fold_metrics = [], [], [], [], []
    for task in TASKS:
        observed = data.loc[data[task].notna(), ["smiles", "scaffold", task]].reset_index(drop=True)
        observed = observed.rename(columns={task: "y"})
        groups = observed.scaffold.astype(str).to_numpy()
        outer_splits = get_folds(observed, groups, task, args.outer_folds, args.folds_file)
        ids = observed.smiles.astype(str).map(feature_map).to_numpy(dtype=int)
        x = x2[ids]
        y = observed.y.to_numpy(dtype=np.float64)
        for fold, (outer_train, outer_test) in enumerate(outer_splits):
            inner = GroupShuffleSplit(n_splits=1, test_size=0.12,
                                      random_state=args.seed + fold)
            fit_rel, val_rel = next(inner.split(outer_train, y[outer_train], groups[outer_train]))
            fit_idx, val_idx = outer_train[fit_rel], outer_train[val_rel]
            if (set(groups[fit_idx]) & set(groups[val_idx])
                    or set(groups[outer_train]) & set(groups[outer_test])):
                raise RuntimeError(f"Scaffold leakage: {task}/fold{fold}")
            candidates = []
            for name, prototype in base.candidates("2d", args.seed + fold, args.n_estimators):
                model = prototype
                if hasattr(model, "n_jobs"):
                    model.set_params(n_jobs=1)
                model.fit(x[fit_idx], y[fit_idx])
                val_pred = np.asarray(model.predict(x[val_idx])).reshape(-1)
                test_pred = np.asarray(model.predict(x[outer_test])).reshape(-1)
                score = float(np.sqrt(mean_squared_error(y[val_idx], val_pred)))
                scores.append({"task": task, "outer_fold": fold, "candidate": name,
                               "inner_validation_rmse": score,
                               "n_fit": len(fit_idx), "n_validation": len(val_idx)})
                candidates.append((score, name, test_pred, model))
            score, name, test_pred, selected_model = min(candidates, key=lambda row: row[0])
            full_model = clone(selected_model)
            full_model.fit(x[outer_train], y[outer_train])
            full_test_pred = np.asarray(full_model.predict(x[outer_test])).reshape(-1)
            selections.append({"task": task, "outer_fold": fold, "selected_candidate": name,
                               "inner_validation_rmse": score,
                               "n_fit": len(fit_idx), "n_validation": len(val_idx),
                               "n_outer_test": len(outer_test)})
            for pos, row_idx in enumerate(outer_test):
                common = {"task": task, "outer_fold": fold,
                          "smiles": str(observed.iloc[row_idx].smiles),
                          "scaffold": str(groups[row_idx]), "y_true": float(y[row_idx])}
                predictions.append({**common, "y_pred": float(test_pred[pos]),
                                    "model": "2d_matched_inner_train_selected"})
                full_predictions.append({**common, "y_pred": float(full_test_pred[pos]),
                                        "model": "2d_vector_fulltrain_inner_selected"})
            rmse = float(np.sqrt(mean_squared_error(y[outer_test], test_pred)))
            fold_metrics.append({"task": task, "outer_fold": fold, "rmse": rmse,
                                 "n_test": len(outer_test), "selected_candidate": name,
                                 "inner_validation_rmse": score})
            pd.DataFrame(scores).to_csv(out / "inner_candidate_scores.csv", index=False)
            pd.DataFrame(selections).to_csv(out / "inner_model_selections.csv", index=False)
            pd.DataFrame(predictions).to_csv(out / "outer_oof_predictions.csv", index=False)
            pd.DataFrame(full_predictions).to_csv(out / "outer_oof_predictions_fulltrain.csv", index=False)
            pd.DataFrame(fold_metrics).to_csv(out / "outer_fold_metrics.csv", index=False)
            print(f"{task} fold={fold + 1}/{args.outer_folds} selected={name} "
                  f"inner_rmse={score:.4f} outer_rmse={rmse:.4f}", flush=True)

    pd.DataFrame(scores).to_csv(out / "inner_candidate_scores.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "inner_model_selections.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "outer_oof_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(out / "outer_fold_metrics.csv", index=False)
    audit = {
        "dataset_sha256": sha256(DATA), "feature_sha256": sha256(FEATURES),
        "outer_fold_manifest": args.folds_file, "inner_split_seed": args.seed,
        "inner_validation_fraction": 0.12, "fold_count_per_task": args.outer_folds,
        "selection_uses_outer_test": False,
        "fit_fraction_matches_unimol2": "same 2D vector candidate family, GroupShuffleSplit seed, and 12% inner scaffold validation",
        "zero_scaffold_leakage": True,
        "full_training_refit_model": "same candidate selected on inner validation, refit on complete outer training data",
        "predictions_per_task": {task: int(sum(r["task"] == task for r in predictions)) for task in TASKS},
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
