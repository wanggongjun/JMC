"""Paired nested scaffold CV for 3D shape descriptors and 2D+3D features.

Outer scaffold assignments and the 2D-only predictions are read from the
frozen v1 nested benchmark. Candidate selection for shape-only and augmented
models is performed with inner scaffold folds only.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/modeling_dataset.csv"
FEAT2 = ROOT / "results/2d_morgan_rdkit_features.npz"
FEAT3 = ROOT / "results/rdkit_3d_shape_ensemble.npz"
BASELINE = ROOT / "experiments/optimization_v1/nested_cv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"n": int(len(y)), "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}


def candidates(mode: str, seed: int):
    out = []
    if mode == "shape3d":
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            out.append((f"ridge_alpha{alpha:g}", make_pipeline(
                StandardScaler(), Ridge(alpha=alpha, solver="lsqr", max_iter=5000, tol=1e-4))))
        for c in (0.1, 1.0, 10.0):
            out.append((f"svr_rbf_c{c:g}", make_pipeline(StandardScaler(), SVR(C=c, epsilon=0.15))))
        for leaf in (2, 8):
            for max_features in (0.8, 1.0):
                out.append((f"extra_trees_leaf{leaf}_mf{max_features:g}", ExtraTreesRegressor(
                    n_estimators=120, min_samples_leaf=leaf, max_features=max_features,
                    n_jobs=4, random_state=seed)))
    elif mode == "morgan_plus_shape3d":
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            out.append((f"ridge_alpha{alpha:g}", make_pipeline(
                StandardScaler(), Ridge(alpha=alpha, solver="lsqr", max_iter=5000, tol=1e-4))))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=("hepg2_pIC50", "hct116_pIC50"),
                        default=["hepg2_pIC50", "hct116_pIC50"])
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--features-3d", default=str(FEAT3.relative_to(ROOT)))
    parser.add_argument("--output-dir", default="experiments/optimization_v1/nested_3d_shape_cv")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    f2 = np.load(FEAT2, allow_pickle=False)
    feature_path = Path(args.features_3d)
    if not feature_path.is_absolute():
        feature_path = ROOT / feature_path
    f3 = np.load(feature_path, allow_pickle=False)
    map2 = {str(s): i for i, s in enumerate(f2["smiles"])}
    map3 = {str(s): i for i, s in enumerate(f3["smiles"])}
    smiles_all = data.smiles.astype(str).tolist()
    if any(s not in map2 for s in smiles_all) or any(s not in map3 for s in smiles_all):
        raise RuntimeError("Feature files do not cover the exact modeling dataset")
    x2_all = f2["features"][np.asarray([map2[s] for s in smiles_all])].astype(np.float32)
    x3_all = f3["features"][np.asarray([map3[s] for s in smiles_all])].astype(np.float32)
    if not np.isfinite(x2_all).all() or not np.isfinite(x3_all).all():
        raise RuntimeError("Non-finite feature values")
    folds_df = pd.read_csv(BASELINE / "outer_scaffold_folds.csv")
    baseline_df = pd.read_csv(BASELINE / "outer_oof_predictions.csv")
    baseline_df = baseline_df.loc[baseline_df.model == "2d_best"].copy()
    baseline_df["scaffold"] = baseline_df["scaffold"].fillna("").astype(str)
    if set(args.tasks) - set(folds_df.task.unique()) or set(args.tasks) - set(baseline_df.task.unique()):
        raise RuntimeError("Missing baseline scaffold assignments or predictions")

    candidate_rows, selection_rows, pred_rows, fold_rows, split_rows = [], [], [], [], []
    for task in args.tasks:
        observed = data.loc[data[task].notna(), ["smiles", task]].reset_index(drop=True)
        smi = observed.smiles.astype(str).to_numpy()
        ids = np.asarray([data.index[data.smiles.astype(str) == s][0] for s in smi], dtype=int)
        y = observed[task].to_numpy(dtype=np.float32)
        groups = np.asarray([scaffold(s) for s in smi])
        assignment = folds_df.loc[folds_df.task == task].set_index("smiles")["outer_fold"].to_dict()
        if set(smi) != set(assignment):
            raise RuntimeError(f"Scaffold assignment mismatch for {task}")
        outer_ids = np.asarray([int(assignment[s]) for s in smi], dtype=int)
        base = baseline_df.loc[baseline_df.task == task].set_index(["outer_fold", "smiles"])
        expected_keys = {(int(outer_ids[i]), smi[i]) for i in range(len(smi))}
        got_keys = set((int(k[0]), str(k[1])) for k in base.index)
        if got_keys != expected_keys:
            raise RuntimeError(f"Baseline prediction coverage mismatch for {task}")
        modes = {
            "shape3d_best": x3_all[ids],
            "morgan_plus_shape3d_best": np.concatenate([x2_all[ids], x3_all[ids]], axis=1),
        }
        for outer_fold in sorted(np.unique(outer_ids)):
            tr = np.flatnonzero(outer_ids != outer_fold)
            te = np.flatnonzero(outer_ids == outer_fold)
            if set(groups[tr]) & set(groups[te]):
                raise RuntimeError(f"Outer scaffold leakage {task}/{outer_fold}")
            inner = GroupKFold(n_splits=args.inner_folds)
            inner_splits = list(inner.split(tr, y[tr], groups[tr]))
            for model_label, X in modes.items():
                mode = "shape3d" if model_label == "shape3d_best" else "morgan_plus_shape3d"
                scored = []
                for name, proto in candidates(mode, 20260925 + outer_fold):
                    oof = np.full(len(tr), np.nan, dtype=np.float32)
                    for inner_i, (fit_rel, val_rel) in enumerate(inner_splits):
                        fit_ids, val_ids = tr[fit_rel], tr[val_rel]
                        model = clone(proto)
                        if hasattr(model, "random_state"):
                            model.set_params(random_state=20260925 + outer_fold * 100 + inner_i)
                        with threadpool_limits(limits=1, user_api="blas"):
                            model.fit(X[fit_ids], y[fit_ids])
                            oof[val_rel] = model.predict(X[val_ids])
                    if not np.isfinite(oof).all():
                        raise RuntimeError(f"Incomplete inner OOF: {task}/{outer_fold}/{name}")
                    score = float(np.sqrt(mean_squared_error(y[tr], oof)))
                    candidate_rows.append({"task": task, "outer_fold": int(outer_fold),
                                           "model": model_label, "candidate": name,
                                           "inner_oof_rmse": score})
                    scored.append((score, name, proto))
                    print(f"{task} outer={outer_fold+1}/5 {model_label} {name} inner_rmse={score:.4f}", flush=True)
                score, name, proto = min(scored, key=lambda z: z[0])
                fitted = clone(proto)
                if hasattr(fitted, "random_state"):
                    fitted.set_params(random_state=20260925 + outer_fold)
                with threadpool_limits(limits=1, user_api="blas"):
                    fitted.fit(X[tr], y[tr])
                    pred = fitted.predict(X[te]).astype(float)
                selection_rows.append({"task": task, "outer_fold": int(outer_fold),
                                       "model": model_label, "selected_candidate": name,
                                       "inner_oof_rmse": score})
                fold_metric = metric(y[te], pred)
                fold_rows.append({"task": task, "outer_fold": int(outer_fold),
                                  "model": model_label, **fold_metric})
                for pos, row_i in enumerate(te):
                    pred_rows.append({"task": task, "outer_fold": int(outer_fold),
                                      "smiles": smi[row_i], "scaffold": groups[row_i],
                                      "y_true": float(y[row_i]), "model": model_label,
                                      "y_pred": float(pred[pos])})
                    split_rows.append({"task": task, "outer_fold": int(outer_fold),
                                       "smiles": smi[row_i], "scaffold": groups[row_i]})
            # Preserve completed outer folds if a later aggregation/audit check fails.
            pd.DataFrame(pred_rows).to_csv(out_dir / "outer_oof_predictions.csv", index=False)
            pd.DataFrame(candidate_rows).to_csv(out_dir / "inner_candidate_scores.csv", index=False)
            pd.DataFrame(selection_rows).to_csv(out_dir / "inner_model_selections.csv", index=False)

    preds = pd.DataFrame(pred_rows)
    base = baseline_df[["task", "outer_fold", "smiles", "scaffold", "y_true", "y_pred"]].rename(
        columns={"y_pred": "baseline_y_pred"})
    summary_rows, bootstrap_rows, paired_rows = [], [], []
    rng = np.random.default_rng(20260925)
    for (task, model_name), part in preds.groupby(["task", "model"]):
        paired = part.merge(base, on=["task", "outer_fold", "smiles", "scaffold"],
                            how="inner", suffixes=("_candidate", "_baseline"),
                            validate="one_to_one")
        if len(paired) != len(part):
            raise RuntimeError(f"Could not pair all predictions for {task}/{model_name}")
        if not np.allclose(paired.y_true_candidate, paired.y_true_baseline, atol=1e-7, rtol=0):
            raise RuntimeError(f"Label mismatch after prediction pairing for {task}/{model_name}")
        paired["y_true"] = paired["y_true_candidate"]
        m2 = metric(paired.y_true.to_numpy(), paired.baseline_y_pred.to_numpy())
        m3 = metric(paired.y_true.to_numpy(), paired.y_pred.to_numpy())
        scaffolds = paired.scaffold.unique()
        by_scaf = {s: np.flatnonzero(paired.scaffold.to_numpy() == s) for s in scaffolds}
        deltas = []
        for _ in range(2000):
            sampled = rng.choice(scaffolds, size=len(scaffolds), replace=True)
            b = np.concatenate([by_scaf[s] for s in sampled])
            deltas.append(np.sqrt(np.mean((paired.y_pred.to_numpy()[b] - paired.y_true.to_numpy()[b]) ** 2))
                          - np.sqrt(np.mean((paired.baseline_y_pred.to_numpy()[b] - paired.y_true.to_numpy()[b]) ** 2)))
        low, high = np.quantile(deltas, [0.025, 0.975])
        fold_model = preds.loc[(preds.task == task) & (preds.model == model_name)]
        fold_base = base.loc[base.task == task]
        fold_join = fold_model.groupby("outer_fold").apply(lambda g: metric(g.y_true.to_numpy(), g.y_pred.to_numpy())["rmse"])
        fold_base_rmse = fold_base.groupby("outer_fold").apply(lambda g: metric(g.y_true.to_numpy(), g.baseline_y_pred.to_numpy())["rmse"])
        summary_rows.append({"task": task, "model": model_name, "n": len(paired),
                             "rmse_2d_baseline": m2["rmse"], "rmse_candidate": m3["rmse"],
                             "delta_candidate_minus_2d": m3["rmse"] - m2["rmse"],
                             "relative_rmse_improvement_pct": 100 * (m2["rmse"] - m3["rmse"]) / m2["rmse"],
                             "folds_won": int((fold_join < fold_base_rmse).sum()),
                             "scaffold_bootstrap_ci95_low": float(low),
                             "scaffold_bootstrap_ci95_high": float(high),
                             "bootstrap_probability_candidate_better": float(np.mean(np.asarray(deltas) < 0))})
        paired_rows.append(paired.assign(candidate=model_name))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidate_rows).to_csv(out_dir / "inner_candidate_scores.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out_dir / "inner_model_selections.csv", index=False)
    preds.to_csv(out_dir / "outer_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "paired_oof_summary.csv", index=False)
    audit = {"status": "complete",
             "validation": "Paired outer 5-fold Bemis-Murcko GroupKFold; inner 4-fold scaffold selection",
             "outer_scaffold_folds_reused_from": str((BASELINE / "outer_scaffold_folds.csv").relative_to(ROOT)),
             "2d_baseline_predictions": str((BASELINE / "outer_oof_predictions.csv").relative_to(ROOT)),
             "shape_descriptor_file": str(feature_path.relative_to(ROOT) if feature_path.is_relative_to(ROOT) else feature_path),
             "dataset_sha256": sha256(DATA), "features_2d_sha256": sha256(FEAT2),
             "features_3d_sha256": sha256(feature_path), "tasks": args.tasks,
             "warning": "Internal ChEMBL scaffold CV; no external prospective validation."}
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
