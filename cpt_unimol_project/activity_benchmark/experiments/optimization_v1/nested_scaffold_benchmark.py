"""Nested scaffold-CV comparison of 2D, frozen Uni-Mol 3D, and fusion.

All hyperparameter selection is confined to inner scaffold folds. Outer folds
produce paired out-of-fold predictions and are never used to select parameters.
Both concatenated-feature and prediction-level fusion are evaluated.
This is internal nested CV on the existing ChEMBL collection, not external
validation. See LITERATURE_AND_PROTOCOL.md before interpreting results.
"""
from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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


def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_candidates(modality: str, seed: int):
    candidates = []
    if modality == "2d":
        for leaf in (2, 8, 16):
            candidates.append((f"extra_trees_leaf{leaf}", ExtraTreesRegressor(
                n_estimators=160, min_samples_leaf=leaf, max_features="sqrt",
                n_jobs=4, random_state=seed,
            ), "features"))
        for leaf in (2, 8, 16):
            candidates.append((f"random_forest_leaf{leaf}", RandomForestRegressor(
                n_estimators=160, min_samples_leaf=leaf, max_features="sqrt",
                n_jobs=4, random_state=seed,
            ), "features"))
        for alpha in (0.01, 0.1, 1.0, 10.0):
            candidates.append((f"tanimoto_krr_alpha{alpha:g}", KernelRidge(
                alpha=alpha, kernel="precomputed"
            ), "tanimoto"))
    elif modality == "3d":
        for leaf in (2, 8, 16):
            candidates.append((f"extra_trees_leaf{leaf}", ExtraTreesRegressor(
                n_estimators=160, min_samples_leaf=leaf, max_features="sqrt",
                n_jobs=4, random_state=seed,
            ), "features"))
        for leaf in (2, 8, 16):
            candidates.append((f"random_forest_leaf{leaf}", RandomForestRegressor(
                n_estimators=160, min_samples_leaf=leaf, max_features="sqrt",
                n_jobs=4, random_state=seed,
            ), "features"))
        for c in (0.1, 1.0, 10.0):
            candidates.append((f"svr_rbf_c{c:g}", make_pipeline(
                StandardScaler(), SVR(C=c, epsilon=0.15, gamma="scale")
            ), "features"))
    elif modality == "fused":
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            candidates.append((f"ridge_alpha{alpha:g}", make_pipeline(
                StandardScaler(), Ridge(alpha=alpha)
            ), "features"))
        for c in (0.1, 1.0, 10.0):
            candidates.append((f"svr_rbf_c{c:g}", make_pipeline(
                StandardScaler(), SVR(C=c, epsilon=0.15, gamma="scale")
            ), "features"))
        for leaf in (2, 8):
            for max_features in (0.05, 0.15):
                candidates.append((f"extra_trees_leaf{leaf}_mf{max_features:g}", ExtraTreesRegressor(
                    n_estimators=160, min_samples_leaf=leaf, max_features=max_features,
                    n_jobs=4, random_state=seed,
                ), "features"))
    else:
        raise ValueError(modality)
    return candidates


def predict_with(name, model, kind, X, K, train_idx, eval_idx):
    if kind == "tanimoto":
        model.fit(K[np.ix_(train_idx, train_idx)], y_global[train_idx])
        return model.predict(K[np.ix_(eval_idx, train_idx)])
    fit_x = X[train_idx]
    eval_x = X[eval_idx]
    model.fit(fit_x, y_global[train_idx])
    pred = model.predict(eval_x)
    if not np.isfinite(pred).all():
        raise FloatingPointError(f"Non-finite predictions from {name}")
    return pred


def metrics(y, pred):
    pearson = float(np.corrcoef(y, pred)[0, 1]) if np.std(y) > 0 and np.std(pred) > 0 else 0.0
    spearman = float(pd.Series(y).corr(pd.Series(pred), method="spearman"))
    if not np.isfinite(spearman):
        spearman = 0.0
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "pearson": pearson,
        "spearman": spearman,
    }


def main():
    global y_global
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-3d", default="results/unimol_v1_cls_repr.npz")
    parser.add_argument("--output-dir", default="experiments/optimization_v1/nested_cv")
    parser.add_argument("--tasks", nargs="+", choices=("hepg2_pIC50", "hct116_pIC50"),
                        default=["hepg2_pIC50", "hct116_pIC50"])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--limit-rows", type=int, default=0, help="Smoke-test row limit")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    feat3_path = Path(args.features_3d)
    if not feat3_path.is_absolute():
        feat3_path = ROOT / feat3_path
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("outer and inner folds must both be >= 2")
    data = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    if args.limit_rows:
        data = data.head(args.limit_rows).copy().reset_index(drop=True)
    f2 = np.load(FEAT2, allow_pickle=False)
    f3 = np.load(feat3_path, allow_pickle=False)
    smiles2 = f2["smiles"].astype(str).tolist()
    smiles3 = f3["smiles"].astype(str).tolist()
    smiles_all = data.smiles.astype(str).tolist()
    map2 = {s: i for i, s in enumerate(smiles2)}
    map3 = {s: i for i, s in enumerate(smiles3)}
    missing2 = [s for s in smiles_all if s not in map2]
    missing3 = [s for s in smiles_all if s not in map3]
    if missing2 or missing3:
        raise RuntimeError(f"Feature mismatch: missing 2D={len(missing2)}, missing 3D={len(missing3)}")
    X2 = f2["features"][np.asarray([map2[s] for s in smiles_all])].astype(np.float32)
    X3 = f3["cls_repr"][np.asarray([map3[s] for s in smiles_all])].astype(np.float32)
    Morgan = X2[:, :2048].astype(np.float64)
    with threadpool_limits(limits=1, user_api="blas"), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        dot = Morgan @ Morgan.T
    norms = Morgan.sum(axis=1, keepdims=True)
    denom = norms + norms.T - dot
    K = np.divide(dot, denom, out=np.zeros_like(dot, dtype=np.float64), where=denom > 0).astype(np.float32)
    del Morgan, dot, norms, denom
    all_row = {s: i for i, s in enumerate(smiles_all)}
    outer_rows, candidate_rows, prediction_rows, selection_rows = [], [], [], []
    split_rows = []

    for task in args.tasks:
        observed = data.loc[data[task].notna(), ["smiles", task]].copy().reset_index(drop=True)
        ids = np.asarray([all_row[s] for s in observed.smiles.astype(str)])
        y_global = observed[task].to_numpy(dtype=np.float32)
        groups = np.asarray([get_scaffold(s) for s in observed.smiles.astype(str)])
        outer = GroupKFold(n_splits=args.outer_folds)
        outer_pred = {"2d": np.full(len(observed), np.nan), "3d": np.full(len(observed), np.nan),
                      "fused": np.full(len(observed), np.nan), "hybrid": np.full(len(observed), np.nan)}
        outer_fold_ids = np.full(len(observed), -1, dtype=int)
        for outer_fold, (tr_local, te_local) in enumerate(outer.split(np.arange(len(observed)), y_global, groups)):
            outer_fold_ids[te_local] = outer_fold
            X2_task, X3_task, K_task = X2[ids], X3[ids], K[np.ix_(ids, ids)]
            groups_train = groups[tr_local]
            inner = GroupKFold(n_splits=args.inner_folds)
            inner_splits = list(inner.split(np.arange(len(tr_local)), y_global[tr_local], groups_train))
            best = {}
            oof_selected = {}
            Xf_task = np.concatenate([X2_task, X3_task], axis=1)
            for modality, X in (("2d", X2_task), ("3d", X3_task), ("fused", Xf_task)):
                scored = []
                candidates = make_candidates(modality, seed=20260925 + outer_fold)
                for name, proto, kind in candidates:
                    oof = np.full(len(tr_local), np.nan, dtype=np.float32)
                    for inner_fold, (in_tr_local, in_va_local) in enumerate(inner_splits):
                        fit_ids = tr_local[in_tr_local]
                        val_ids = tr_local[in_va_local]
                        # Clone through sklearn to avoid carrying fitted state between folds.
                        from sklearn.base import clone
                        model = clone(proto)
                        rngseed = 20260925 + outer_fold * 1000 + inner_fold
                        if hasattr(model, "random_state"):
                            model.set_params(random_state=rngseed)
                        pred = predict_with(name, model, kind, X, K_task, fit_ids, val_ids)
                        oof[in_va_local] = pred
                    if not np.isfinite(oof).all():
                        raise RuntimeError(f"Incomplete inner OOF predictions: {task}/{name}")
                    score = float(np.sqrt(mean_squared_error(y_global[tr_local], oof)))
                    candidate_rows.append({"task": task, "outer_fold": outer_fold, "modality": modality,
                                           "model": name, "inner_oof_rmse": score})
                    scored.append((score, name, proto, kind, oof))
                    print(f"{task} outer={outer_fold+1}/{args.outer_folds} {modality} candidate={name} inner_rmse={score:.4f}", flush=True)
                score, name, proto, kind, oof = min(scored, key=lambda item: item[0])
                best[modality] = (score, name, proto, kind)
                oof_selected[modality] = oof
                from sklearn.base import clone
                final_model = clone(proto)
                if hasattr(final_model, "random_state"):
                    final_model.set_params(random_state=20260925 + outer_fold)
                pred = predict_with(name, final_model, kind, X, K_task, tr_local, te_local)
                outer_pred[modality][te_local] = pred
                selection_rows.append({"task": task, "outer_fold": outer_fold,
                                       "modality": modality, "selected_model": name,
                                       "selected_2d_model": "" if modality != "2d" else name,
                                       "selected_3d_model": "" if modality != "3d" else name,
                                       "selected_inner_oof_rmse": score})
                print(f"{task} outer={outer_fold+1}/{args.outer_folds} {modality} selected={name} inner_rmse={score:.4f}", flush=True)

            # The blend is learned from inner OOF predictions only. alpha is the 3D weight.
            best_alpha, best_inner_rmse = 0.0, float("inf")
            oof2, oof3 = oof_selected["2d"], oof_selected["3d"]
            for alpha in np.linspace(0.0, 1.0, 21):
                mixed = (1.0 - alpha) * oof2 + alpha * oof3
                score = float(np.sqrt(mean_squared_error(y_global[tr_local], mixed)))
                if score < best_inner_rmse:
                    best_alpha, best_inner_rmse = float(alpha), score
            outer_pred["hybrid"][te_local] = (1.0 - best_alpha) * outer_pred["2d"][te_local] + best_alpha * outer_pred["3d"][te_local]
            selection_rows.append({"task": task, "outer_fold": outer_fold,
                                   "selected_2d_model": best["2d"][1],
                                   "selected_3d_model": best["3d"][1],
                                   "selected_inner_oof_rmse": best_inner_rmse,
                                   "hybrid_3d_weight": best_alpha})
            for name, pred in (("2d_best", outer_pred["2d"][te_local]),
                               ("3d_best", outer_pred["3d"][te_local]),
                               ("2d_3d_early_fusion", outer_pred["fused"][te_local]),
                               ("2d_3d_hybrid", outer_pred["hybrid"][te_local])):
                score = metrics(y_global[te_local], pred)
                outer_rows.append({"task": task, "outer_fold": outer_fold, "model": name,
                                   "selected_2d_model": best["2d"][1],
                                   "selected_3d_model": best["3d"][1],
                                   "hybrid_3d_weight": best_alpha, **score})
            for idx in te_local:
                for name, key in (("2d_best", "2d"), ("3d_best", "3d"),
                                  ("2d_3d_early_fusion", "fused"), ("2d_3d_hybrid", "hybrid")):
                    prediction_rows.append({"task": task, "outer_fold": int(outer_fold), "smiles": observed.iloc[idx].smiles,
                                            "scaffold": groups[idx], "y_true": float(y_global[idx]),
                                            "model": name, "y_pred": float(outer_pred[key][idx])})
            split_rows.extend({"task": task, "smiles": observed.iloc[i].smiles,
                               "scaffold": groups[i], "outer_fold": int(outer_fold)} for i in te_local)

        for name, pred in (("2d_best", outer_pred["2d"]), ("3d_best", outer_pred["3d"]),
                           ("2d_3d_early_fusion", outer_pred["fused"]),
                           ("2d_3d_hybrid", outer_pred["hybrid"])):
            if not np.isfinite(pred).all():
                raise RuntimeError(f"Outer CV left unpredicted rows for {task}/{name}")

    pd.DataFrame(outer_rows).to_csv(out_dir / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(out_dir / "inner_candidate_scores.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out_dir / "outer_oof_predictions.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out_dir / "inner_model_selections.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out_dir / "outer_scaffold_folds.csv", index=False)

    preds = pd.DataFrame(prediction_rows)
    pooled = []
    for (task, name), part in preds.groupby(["task", "model"]):
        pooled.append({"task": task, "model": name, **metrics(part.y_true.to_numpy(), part.y_pred.to_numpy())})
    pd.DataFrame(pooled).to_csv(out_dir / "pooled_nested_cv_metrics.csv", index=False)
    audit = {"status": "complete", "outer_method": f"{args.outer_folds}-fold Bemis-Murcko GroupKFold",
             "inner_method": f"{args.inner_folds}-fold Bemis-Murcko GroupKFold",
             "outer_test_predictions_per_task": {task: int((preds.task == task).sum() / 4) for task in args.tasks},
             "features": {"2d": list(X2.shape), "3d": list(X3.shape),
                          "early_fusion_2d_plus_3d": [int(X2.shape[0]), int(X2.shape[1] + X3.shape[1])]},
             "features_3d_path": str(feat3_path),
             "dataset_sha256": sha256(DATA),
             "features_2d_sha256": sha256(FEAT2),
             "features_3d_sha256": sha256(feat3_path),
             "tasks": args.tasks,
             "smoke_test_limit_rows": args.limit_rows or None,
             "production_result": not bool(args.limit_rows),
             "scaffold_overlap": 0,
             "warning": "Nested CV is internal validation on a ChEMBL dataset previously used in an exploratory benchmark; it is not external validation."}
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(pooled).to_string(index=False), flush=True)


if __name__ == "__main__":
    y_global = np.array([], dtype=np.float32)
    main()
