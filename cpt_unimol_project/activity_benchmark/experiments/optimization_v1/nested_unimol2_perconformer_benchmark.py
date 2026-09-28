"""Molecule-grouped nonaggregation benchmark using each Uni-Mol2 conformer.

Each conformer is a supervised training instance with its molecule's label;
all conformers of a molecule stay in the same inner/outer split. At validation
and test time, per-conformer predictions are averaged back to one molecule.
This is the frozen-embedding counterpart to conformer-augmentation training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/modeling_dataset.csv"
FEAT2 = ROOT / "results/2d_morgan_rdkit_features.npz"
BASELINE = ROOT / "experiments/optimization_v1/nested_cv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def model_candidates(mode: str, seed: int):
    out = []
    if mode == "3d":
        for leaf in (2, 8, 16):
            out.append((f"extra_trees_leaf{leaf}", ExtraTreesRegressor(
                n_estimators=120, min_samples_leaf=leaf, max_features="sqrt",
                n_jobs=4, random_state=seed)))
        for leaf in (2, 8, 16):
            out.append((f"random_forest_leaf{leaf}", RandomForestRegressor(
                n_estimators=120, min_samples_leaf=leaf, max_features="sqrt",
                n_jobs=4, random_state=seed)))
        for c in (0.1, 1.0, 10.0):
            out.append((f"svr_rbf_c{c:g}", make_pipeline(StandardScaler(), SVR(C=c, epsilon=0.15))))
    elif mode == "fused":
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            out.append((f"ridge_alpha{alpha:g}", make_pipeline(StandardScaler(), Ridge(alpha=alpha))))
        for c in (0.1, 1.0, 10.0):
            out.append((f"svr_rbf_c{c:g}", make_pipeline(StandardScaler(), SVR(C=c, epsilon=0.15))))
        for leaf in (2, 8):
            for max_features in (0.05, 0.15):
                out.append((f"extra_trees_leaf{leaf}_mf{max_features:g}", ExtraTreesRegressor(
                    n_estimators=120, min_samples_leaf=leaf, max_features=max_features,
                    n_jobs=4, random_state=seed)))
    else:
        raise ValueError(mode)
    return out


def expand_rows(molecule_ids: np.ndarray, per_conf: np.ndarray, mask: np.ndarray,
                x2: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    counts = mask[molecule_ids].sum(axis=1).astype(int)
    parent = np.repeat(molecule_ids, counts)
    reps = per_conf[molecule_ids][mask[molecule_ids]]
    if x2 is not None:
        reps = np.concatenate([x2[parent], reps], axis=1)
    return reps.astype(np.float32), parent


def aggregate_predictions(conf_preds: np.ndarray, parent_ids: np.ndarray,
                          target_ids: np.ndarray) -> np.ndarray:
    loc = {int(g): i for i, g in enumerate(target_ids)}
    positions = np.asarray([loc[int(g)] for g in parent_ids], dtype=np.int64)
    sums = np.zeros(len(target_ids), dtype=np.float64)
    counts = np.zeros(len(target_ids), dtype=np.int32)
    np.add.at(sums, positions, conf_preds.astype(np.float64))
    np.add.at(counts, positions, 1)
    if np.any(counts == 0):
        raise RuntimeError("A molecule has no conformer prediction")
    return (sums / counts).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-3d", default="results/unimol2_84m_ensemble_perconf_repr.npz")
    parser.add_argument("--tasks", nargs="+", choices=("hepg2_pIC50", "hct116_pIC50"),
                        default=["hepg2_pIC50", "hct116_pIC50"])
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--max-outer-folds", type=int, default=0, help="Debug/smoke run only")
    parser.add_argument("--output-dir", default="experiments/optimization_v1/nested_unimol2_perconformer_cv")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = Path(args.features_3d)
    if not feature_path.is_absolute():
        feature_path = ROOT / feature_path

    data = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    feat2 = np.load(FEAT2, allow_pickle=False)
    feat3 = np.load(feature_path, allow_pickle=False)
    map2 = {str(s): i for i, s in enumerate(feat2["smiles"])}
    map3 = {str(s): i for i, s in enumerate(feat3["smiles"])}
    smiles_all = data.smiles.astype(str).tolist()
    if set(smiles_all) != set(map2) or set(smiles_all) != set(map3):
        raise RuntimeError("SMILES coverage does not exactly match the modeling dataset")
    x2_all = feat2["features"][np.asarray([map2[s] for s in smiles_all])].astype(np.float32)
    per_all = feat3["per_conf_repr"][np.asarray([map3[s] for s in smiles_all])].astype(np.float32)
    mask_all = feat3["conformer_mask"][np.asarray([map3[s] for s in smiles_all])].astype(bool)
    if per_all.ndim != 3 or mask_all.shape != per_all.shape[:2] or not mask_all.any(axis=1).all():
        raise RuntimeError("Invalid per-conformer embeddings or masks")
    if not np.isfinite(per_all[mask_all]).all() or not np.isfinite(x2_all).all():
        raise RuntimeError("Non-finite molecular features")

    fold_table = pd.read_csv(BASELINE / "outer_scaffold_folds.csv")
    baseline = pd.read_csv(BASELINE / "outer_oof_predictions.csv")
    baseline = baseline.loc[baseline.model == "2d_best"].copy()
    prediction_path = output_dir / "outer_oof_predictions.csv"
    candidate_path = output_dir / "inner_candidate_scores.csv"
    selection_path = output_dir / "inner_model_selections.csv"
    prediction_rows = pd.read_csv(prediction_path).to_dict(orient="records") if prediction_path.exists() else []
    candidate_rows = pd.read_csv(candidate_path).to_dict(orient="records") if candidate_path.exists() else []
    selection_rows = pd.read_csv(selection_path).to_dict(orient="records") if selection_path.exists() else []
    completed = set()
    if selection_rows:
        prior_selections = pd.DataFrame(selection_rows)
        for (prior_task, prior_fold), part in prior_selections.groupby(["task", "outer_fold"]):
            if set(part.model.astype(str)) == {"unimol2_perconformer_3d", "unimol2_perconformer_2d3d"}:
                completed.add((str(prior_task), int(prior_fold)))
    for task in args.tasks:
        observed = data.loc[data[task].notna(), ["smiles", task]].reset_index(drop=True)
        smiles = observed.smiles.astype(str).to_numpy()
        ids = np.asarray([data.index[data.smiles.astype(str) == s][0] for s in smiles], dtype=int)
        y = observed[task].to_numpy(dtype=np.float32)
        groups = np.asarray([get_scaffold(s) for s in smiles])
        assignment = fold_table.loc[fold_table.task == task].set_index("smiles").outer_fold.to_dict()
        if set(smiles) != set(assignment):
            raise RuntimeError(f"Outer-fold assignment mismatch for {task}")
        fold_ids = np.asarray([int(assignment[s]) for s in smiles])
        folds = sorted(np.unique(fold_ids))
        if args.max_outer_folds:
            folds = folds[:args.max_outer_folds]
        for outer_fold in folds:
            if (task, int(outer_fold)) in completed:
                continue
            tr = np.flatnonzero(fold_ids != outer_fold)
            te = np.flatnonzero(fold_ids == outer_fold)
            if set(groups[tr]) & set(groups[te]):
                raise RuntimeError(f"Scaffold leakage {task}/{outer_fold}")
            inner = GroupKFold(n_splits=args.inner_folds)
            inner_splits = list(inner.split(tr, y[tr], groups[tr]))
            modes = {
                "unimol2_perconformer_3d": ("3d", None),
                "unimol2_perconformer_2d3d": ("fused", x2_all),
            }
            for model_label, (mode, x2_task) in modes.items():
                scored = []
                for name, prototype in model_candidates(mode, 20260925 + int(outer_fold)):
                    oof = np.full(len(tr), np.nan, dtype=np.float32)
                    for inner_i, (fit_rel, val_rel) in enumerate(inner_splits):
                        fit_ids, val_ids = tr[fit_rel], tr[val_rel]
                        xfit, parent_fit = expand_rows(ids[fit_ids], per_all, mask_all, x2_task)
                        xval, parent_val = expand_rows(ids[val_ids], per_all, mask_all, x2_task)
                        model = clone(prototype)
                        if hasattr(model, "random_state"):
                            model.set_params(random_state=20260925 + int(outer_fold) * 100 + inner_i)
                        with threadpool_limits(limits=1, user_api="blas"):
                            model.fit(xfit, y[np.searchsorted(ids, parent_fit)])
                            conf_pred = model.predict(xval)
                        if not np.isfinite(conf_pred).all():
                            raise FloatingPointError(f"Non-finite conformer predictions: {task}/{outer_fold}/{name}")
                        oof[val_rel] = aggregate_predictions(conf_pred, parent_val, ids[val_ids])
                    if not np.isfinite(oof).all():
                        raise RuntimeError(f"Incomplete molecule-level OOF: {task}/{outer_fold}/{name}")
                    score = float(np.sqrt(mean_squared_error(y[tr], oof)))
                    candidate_rows.append({"task": task, "outer_fold": int(outer_fold),
                                           "model": model_label, "candidate": name,
                                           "inner_oof_rmse": score})
                    scored.append((score, name, prototype))
                    print(f"{task} outer={outer_fold+1}/5 {model_label} {name} inner_rmse={score:.4f}", flush=True)
                score, name, prototype = min(scored, key=lambda x: x[0])
                xfit, parent_fit = expand_rows(ids[tr], per_all, mask_all, x2_task)
                xte, parent_te = expand_rows(ids[te], per_all, mask_all, x2_task)
                model = clone(prototype)
                if hasattr(model, "random_state"):
                    model.set_params(random_state=20260925 + int(outer_fold))
                with threadpool_limits(limits=1, user_api="blas"):
                    model.fit(xfit, y[np.searchsorted(ids, parent_fit)])
                    conf_pred = model.predict(xte)
                if not np.isfinite(conf_pred).all():
                    raise FloatingPointError(f"Non-finite outer conformer predictions: {task}/{outer_fold}/{name}")
                pred = aggregate_predictions(conf_pred, parent_te, ids[te])
                selection_rows.append({"task": task, "outer_fold": int(outer_fold),
                                       "model": model_label, "selected_candidate": name,
                                       "inner_oof_rmse": score})
                for j, local_i in enumerate(te):
                    prediction_rows.append({"task": task, "outer_fold": int(outer_fold),
                                            "smiles": smiles[local_i], "scaffold": groups[local_i],
                                            "y_true": float(y[local_i]), "model": model_label,
                                            "y_pred": float(pred[j])})
            # Checkpoint only after both 3D and 2D+3D outputs for this complete
            # molecule/scaffold fold have been written.
            pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
            pd.DataFrame(candidate_rows).to_csv(candidate_path, index=False)
            pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
            completed.add((task, int(outer_fold)))

    preds = pd.DataFrame(prediction_rows)
    preds.to_csv(prediction_path, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidate_path, index=False)
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    base = baseline[["task", "outer_fold", "smiles", "scaffold", "y_true", "y_pred"]].copy()
    preds["scaffold"] = preds["scaffold"].fillna("").astype(str)
    base["scaffold"] = base["scaffold"].fillna("").astype(str)
    paired = preds.merge(base, on=["task", "outer_fold", "smiles", "scaffold"],
                         suffixes=("_3d", "_2d"), validate="many_to_one")
    if len(paired) != len(preds) or not np.allclose(paired.y_true_3d, paired.y_true_2d, atol=1e-7, rtol=0):
        raise RuntimeError("Could not pair per-conformer predictions to the frozen 2D baseline")
    summary = []
    for (task, model_name), part in paired.groupby(["task", "model"]):
        summary.append({"task": task, "model": model_name, "n": len(part),
                        "rmse_2d_baseline": float(np.sqrt(mean_squared_error(part.y_true_2d, part.y_pred_2d))),
                        "rmse_candidate": float(np.sqrt(mean_squared_error(part.y_true_3d, part.y_pred_3d))),
                        "delta_candidate_minus_2d": float(np.sqrt(mean_squared_error(part.y_true_3d, part.y_pred_3d))
                                                           - np.sqrt(mean_squared_error(part.y_true_2d, part.y_pred_2d)))})
    pd.DataFrame(summary).to_csv(output_dir / "paired_oof_summary.csv", index=False)
    audit = {"status": "smoke_only" if args.max_outer_folds else "complete",
             "method": "Per-conformer instance-wrapper: supervised fit on each conformer, molecule-grouped splits, mean prediction per molecule",
             "folds": [int(fold) for fold in folds], "inner_folds": int(args.inner_folds),
             "per_conformer_features": str(feature_path.relative_to(ROOT) if feature_path.is_relative_to(ROOT) else feature_path),
             "features_3d_sha256": sha256(feature_path), "features_2d_sha256": sha256(FEAT2),
             "dataset_sha256": sha256(DATA), "tasks": args.tasks,
             "outer_scaffold_overlap": 0,
             "warning": "Internal scaffold CV on the existing ChEMBL cohort; not external prospective validation."}
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
