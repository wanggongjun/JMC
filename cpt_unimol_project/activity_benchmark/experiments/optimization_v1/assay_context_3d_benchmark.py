"""Assay-aware, scaffold-held-out test of incremental conformer geometry.

The source contains the original exact ChEMBL IC50 records. Records are kept
separate by assay instead of averaging measurements from different assays.
Each model predicts within-assay residuals after subtracting the training-only
assay mean. All observations of a Bemis-Murcko scaffold stay in one split.

Primary comparison: nested 2D-only versus nested 2D+3D, evaluated on the same
known-assay observations. Unknown-assay rows are also predicted using a
training-only global-mean fallback and reported separately.
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/origindata/chembl_hepg2_hct116_ic50_eq_calibrated.csv"
DATA = ROOT / "data/processed/modeling_dataset.csv"
FEAT2 = ROOT / "results/2d_morgan_rdkit_features.npz"
FEAT3 = ROOT / "results/rdkit_rich_3d_ensemble.npz"
FOLDS = ROOT / "experiments/optimization_v1/nested_cv/outer_scaffold_folds.csv"
TASKS = {"HepG2": "hepg2_pIC50", "HCT116": "hct116_pIC50"}
MODEL_2D = "assay_centered_2d"
MODEL_3D = "assay_centered_3d"
MODEL_FUSED = "assay_centered_2d_plus_3d"
MODEL_CONTEXT = "assay_mean_only"


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


def candidates(mode: str, seed: int, n_estimators: int):
    out = []
    for c in (0.1, 1.0, 10.0):
        out.append((f"svr_rbf_c{c:g}", make_pipeline(StandardScaler(), SVR(C=c, epsilon=0.15))))
    for leaves in (7, 15):
        out.append((f"hist_gradient_boosting_leaves{leaves}", HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=leaves,
            l2_regularization=1.0, early_stopping=False, random_state=seed)))
    max_features_values = (0.05, 0.15, 0.4, 0.8) if mode == "3d" else (0.05, 0.15)
    for leaf in (2, 8):
        for max_features in max_features_values:
            out.append((f"extra_trees_leaf{leaf}_mf{max_features:g}", ExtraTreesRegressor(
                n_estimators=n_estimators, min_samples_leaf=leaf, max_features=max_features,
                n_jobs=4, random_state=seed)))
    return out


def assay_centered_fit_predict(prototype, x: np.ndarray, frame: pd.DataFrame,
                               fit_idx: np.ndarray, pred_idx: np.ndarray,
                               seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = frame.iloc[fit_idx]
    pred = frame.iloc[pred_idx]
    means = fit.groupby("assay_chembl_id").y.mean()
    global_mean = float(fit.y.mean())
    y_centered = fit.y.to_numpy(dtype=np.float64) - fit.assay_chembl_id.map(means).to_numpy(dtype=np.float64)
    model = clone(prototype)
    if "random_state" in model.get_params(deep=False):
        model.set_params(random_state=int(seed))
    with threadpool_limits(limits=1, user_api="blas"):
        model.fit(x[fit_idx], y_centered)
        residual = model.predict(x[pred_idx]).astype(np.float64)
    assay_mean = pred.assay_chembl_id.map(means).to_numpy(dtype=np.float64)
    seen = np.isfinite(assay_mean)
    assay_mean[~seen] = global_mean
    return assay_mean + residual, seen, assay_mean


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"n": int(len(y)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-assay-molecules", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--features-3d", default=str(FEAT3.relative_to(ROOT)))
    parser.add_argument("--feature-key", default=None,
                        help="Array name inside the 3D NPZ; auto-selects `features` or `cls_repr`.")
    parser.add_argument("--max-outer-folds", type=int, default=0, help="Debug only; output audit is smoke_only")
    parser.add_argument("--folds-file", default=str(FOLDS.relative_to(ROOT)))
    parser.add_argument("--output-dir", default="experiments/optimization_v1/assay_context_3d_cv")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
    raw = raw.loc[(raw.standard_type.astype(str).str.upper() == "IC50")
                  & (raw.standard_relation.astype(str) == "=")
                  & raw.pchembl_value_final.notna()].copy()
    raw["task"] = raw.cell_line.map(TASKS)
    raw = raw.dropna(subset=["task", "assay_chembl_id", "canonical_smiles"]).copy()
    raw["smiles"] = raw.canonical_smiles.astype(str)
    raw["y"] = pd.to_numeric(raw.pchembl_value_final, errors="coerce")
    raw = raw.dropna(subset=["y"])
    raw = raw.groupby(["task", "assay_chembl_id", "smiles"], as_index=False).agg(
        y=("y", "mean"), n_source_records=("y", "size"))

    modeling = pd.read_csv(DATA, encoding="utf-8-sig")
    known_smiles = set(modeling.smiles.astype(str))
    raw = raw.loc[raw.smiles.isin(known_smiles)].copy()
    f2 = np.load(FEAT2, allow_pickle=False)
    feature_path = Path(args.features_3d)
    if not feature_path.is_absolute():
        feature_path = ROOT / feature_path
    f3 = np.load(feature_path, allow_pickle=False)
    map2 = {str(s): i for i, s in enumerate(f2["smiles"])}
    map3 = {str(s): i for i, s in enumerate(f3["smiles"])}
    if set(map2) != set(map3):
        raise RuntimeError("2D/3D feature coverage differs")
    feature_key = args.feature_key
    if feature_key is None:
        feature_key = "features" if "features" in f3.files else "cls_repr" if "cls_repr" in f3.files else None
    if feature_key is None or feature_key not in f3.files:
        raise KeyError(f"Select a valid 3D feature key from: {f3.files}")
    raw = raw.loc[raw.smiles.map(lambda s: s in map2 and s in map3)].copy()
    assay_counts = raw.groupby(["task", "assay_chembl_id"]).smiles.nunique().rename("assay_n").reset_index()
    assay_counts["keep"] = assay_counts.assay_n >= args.min_assay_molecules
    kept = assay_counts.loc[assay_counts.keep, ["task", "assay_chembl_id"]]
    raw = raw.merge(kept, on=["task", "assay_chembl_id"], how="inner", validate="many_to_one")

    all_smiles = raw.smiles.drop_duplicates().astype(str).tolist()
    ids_by_smiles = {s: i for i, s in enumerate(all_smiles)}
    x2 = f2["features"][np.asarray([map2[s] for s in all_smiles])].astype(np.float32)
    x3 = f3[feature_key][np.asarray([map3[s] for s in all_smiles])].astype(np.float32)
    if not np.isfinite(x2).all() or not np.isfinite(x3).all():
        raise RuntimeError("Non-finite feature matrix")

    fold_path = Path(args.folds_file)
    if not fold_path.is_absolute():
        fold_path = ROOT / fold_path
    fold_table = pd.read_csv(fold_path)
    fold_maps = {task: dict(zip(part.smiles.astype(str), part.outer_fold.astype(int)))
                 for task, part in fold_table.groupby("task")}
    raw["molecule_id"] = raw.smiles.map(ids_by_smiles).astype(int)
    raw["scaffold"] = raw.smiles.map(scaffold)
    raw["outer_fold"] = [fold_maps[t].get(s, -1) for t, s in zip(raw.task, raw.smiles)]
    if (raw.outer_fold < 0).any():
        raise RuntimeError("One or more assay records lack the frozen scaffold assignment")
    raw = raw.reset_index(drop=True)

    pred_path = out_dir / "outer_oof_predictions.csv"
    cand_path = out_dir / "inner_candidate_scores.csv"
    select_path = out_dir / "inner_model_selections.csv"
    pred_rows = pd.read_csv(pred_path).to_dict("records") if pred_path.exists() else []
    candidate_rows = pd.read_csv(cand_path).to_dict("records") if cand_path.exists() else []
    selection_rows = pd.read_csv(select_path).to_dict("records") if select_path.exists() else []
    completed = set()
    if selection_rows:
        prior = pd.DataFrame(selection_rows)
        for (task, fold), part in prior.groupby(["task", "outer_fold"]):
            if set(part.model.astype(str)) == {MODEL_2D, MODEL_3D, MODEL_FUSED}:
                completed.add((str(task), int(fold)))

    tasks = list(TASKS.values())
    for task in tasks:
        part = raw.loc[raw.task == task].reset_index(drop=True)
        row_folds = part.outer_fold.to_numpy(dtype=int)
        groups = part.scaffold.astype(str).to_numpy()
        folds = sorted(np.unique(row_folds))
        if args.max_outer_folds:
            folds = folds[:args.max_outer_folds]
        for outer_fold in folds:
            if (task, int(outer_fold)) in completed:
                continue
            tr = np.flatnonzero(row_folds != outer_fold)
            te = np.flatnonzero(row_folds == outer_fold)
            if set(groups[tr]) & set(groups[te]):
                raise RuntimeError(f"Outer scaffold leakage in {task}/{outer_fold}")
            inner = GroupKFold(n_splits=args.inner_folds)
            inner_splits = list(inner.split(tr, part.y.to_numpy()[tr], groups[tr]))
            ids = part.molecule_id.to_numpy(dtype=int)
            x_task_2d = x2[ids]
            x_task_3d = x3[ids]
            x_task_fused = np.concatenate([x2[ids], x3[ids]], axis=1)
            matrices = {MODEL_2D: x_task_2d, MODEL_3D: x_task_3d,
                        MODEL_FUSED: x_task_fused}
            prototypes = {MODEL_2D: candidates("2d", 20260925 + outer_fold, args.n_estimators),
                          MODEL_3D: candidates("3d", 20260925 + outer_fold, args.n_estimators),
                          MODEL_FUSED: candidates("2d3d", 20260925 + outer_fold, args.n_estimators)}

            # Assay-mean-only reference uses training rows only.
            fit_assay_means = part.iloc[tr].groupby("assay_chembl_id").y.mean()
            global_mean = float(part.iloc[tr].y.mean())
            test_assay_means = part.iloc[te].assay_chembl_id.map(fit_assay_means)
            seen_context = test_assay_means.notna().to_numpy()
            context_pred = test_assay_means.fillna(global_mean).to_numpy(dtype=float)

            for model_label in (MODEL_2D, MODEL_3D, MODEL_FUSED):
                X = matrices[model_label]
                scored = []
                for name, proto in prototypes[model_label]:
                    oof = np.full(len(tr), np.nan, dtype=np.float64)
                    oof_known_assay = np.zeros(len(tr), dtype=bool)
                    for inner_i, (fit_rel, val_rel) in enumerate(inner_splits):
                        fit_idx, val_idx = tr[fit_rel], tr[val_rel]
                        pred, seen, _ = assay_centered_fit_predict(
                            proto, X, part, fit_idx, val_idx,
                            20260925 + int(outer_fold) * 100 + inner_i)
                        oof[val_rel] = pred
                        oof_known_assay[val_rel] = seen
                    if not np.isfinite(oof).all() or not oof_known_assay.any():
                        raise RuntimeError(f"Incomplete inner predictions for {task}/{outer_fold}/{model_label}/{name}")
                    score = float(np.sqrt(mean_squared_error(
                        part.iloc[tr].y.to_numpy()[oof_known_assay], oof[oof_known_assay])))
                    candidate_rows.append({"task": task, "outer_fold": int(outer_fold),
                                           "model": model_label, "candidate": name,
                                           "inner_oof_rmse": score})
                    scored.append((score, name, proto))
                    print(f"{task} outer={outer_fold + 1}/5 {model_label} {name} inner_rmse={score:.4f}", flush=True)
                score, name, proto = min(scored, key=lambda item: item[0])
                pred, seen, assay_mean = assay_centered_fit_predict(
                    proto, X, part, tr, te, 20260925 + int(outer_fold))
                if not np.array_equal(seen, seen_context):
                    raise RuntimeError("Assay context-support audit mismatch")
                selection_rows.append({"task": task, "outer_fold": int(outer_fold),
                                       "model": model_label, "selected_candidate": name,
                                       "inner_oof_rmse": score})
                for j, idx in enumerate(te):
                    row = part.iloc[idx]
                    pred_rows.append({"task": task, "outer_fold": int(outer_fold),
                                      "assay_chembl_id": str(row.assay_chembl_id),
                                      "smiles": str(row.smiles), "scaffold": str(row.scaffold),
                                      "y_true": float(row.y), "assay_train_n": int((part.iloc[tr].assay_chembl_id == row.assay_chembl_id).sum()),
                                      "assay_train_mean": float(assay_mean[j]),
                                      "assay_seen_in_training": bool(seen[j]),
                                      "model": model_label, "y_pred": float(pred[j])})
            for j, idx in enumerate(te):
                row = part.iloc[idx]
                pred_rows.append({"task": task, "outer_fold": int(outer_fold),
                                  "assay_chembl_id": str(row.assay_chembl_id),
                                  "smiles": str(row.smiles), "scaffold": str(row.scaffold),
                                  "y_true": float(row.y), "assay_train_n": int((part.iloc[tr].assay_chembl_id == row.assay_chembl_id).sum()),
                                  "assay_train_mean": float(context_pred[j]),
                                  "assay_seen_in_training": bool(seen_context[j]),
                                  "model": MODEL_CONTEXT, "y_pred": float(context_pred[j])})
            pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
            pd.DataFrame(candidate_rows).to_csv(cand_path, index=False)
            pd.DataFrame(selection_rows).to_csv(select_path, index=False)
            completed.add((task, int(outer_fold)))

    preds = pd.DataFrame(pred_rows)
    if preds.empty:
        raise RuntimeError("No assay-level predictions were produced")
    summary = []
    pairwise = []
    rng = np.random.default_rng(20260925)
    for task in tasks:
        task_preds = preds.loc[preds.task == task]
        by_model = {name: group.set_index(["outer_fold", "assay_chembl_id", "smiles"])
                    for name, group in task_preds.groupby("model")}
        for model in (MODEL_2D, MODEL_3D, MODEL_FUSED, MODEL_CONTEXT):
            part = task_preds.loc[task_preds.model == model].copy()
            known = part.loc[part.assay_seen_in_training]
            unknown = part.loc[~part.assay_seen_in_training]
            summary.append({"task": task, "model": model, **metric_row(part.y_true.to_numpy(), part.y_pred.to_numpy()),
                            "known_assay_coverage": float(part.assay_seen_in_training.mean()),
                            "known_assay_rmse": metric_row(known.y_true.to_numpy(), known.y_pred.to_numpy())["rmse"],
                            "unseen_assay_rmse": metric_row(unknown.y_true.to_numpy(), unknown.y_pred.to_numpy())["rmse"] if len(unknown) else float("nan"),
                            "macro_assay_rmse": float(part.groupby("assay_chembl_id").apply(
                                lambda g: metric_row(g.y_true.to_numpy(), g.y_pred.to_numpy())["rmse"]).mean()),
                            "known_macro_assay_rmse": float(known.groupby("assay_chembl_id").apply(
                                lambda g: metric_row(g.y_true.to_numpy(), g.y_pred.to_numpy())["rmse"]).mean()),
                            "n_assays_tested": int(part.assay_chembl_id.nunique()),
                            "n_unique_molecules": int(part.smiles.nunique())})
        base_all = by_model[MODEL_2D]
        for model in (MODEL_FUSED, MODEL_3D, MODEL_CONTEXT):
            compare_all = by_model[model]
            base = base_all.loc[base_all.assay_seen_in_training]
            compare = compare_all.loc[compare_all.assay_seen_in_training]
            paired = base[["scaffold", "y_true", "y_pred"]].join(
                compare[["y_pred"]].rename(columns={"y_pred": "candidate_pred"}),
                how="inner", rsuffix="_candidate", validate="one_to_one")
            if len(paired) != len(base) or len(compare) != len(base) or not np.allclose(paired.y_true.to_numpy(),
                                                            compare.loc[base.index, "y_true"].to_numpy(),
                                                            atol=1e-7, rtol=0):
                raise RuntimeError(f"Could not pair exact assay observations for {task}/{model}")
            scaffolds = paired.scaffold.unique()
            by_scaffold = {s: np.flatnonzero(paired.scaffold.to_numpy() == s) for s in scaffolds}
            delta = []
            y = paired.y_true.to_numpy()
            p2 = paired.y_pred.to_numpy()
            p3 = paired.candidate_pred.to_numpy()
            for _ in range(2000):
                sampled = rng.choice(scaffolds, size=len(scaffolds), replace=True)
                idx = np.concatenate([by_scaffold[s] for s in sampled])
                delta.append(float(np.sqrt(np.mean((p3[idx] - y[idx]) ** 2)
                                           - 0.0) - np.sqrt(np.mean((p2[idx] - y[idx]) ** 2))))
            lo, hi = np.quantile(delta, [0.025, 0.975])
            all_paired = base_all[["y_true", "y_pred"]].join(
                compare_all[["y_pred"]].rename(columns={"y_pred": "candidate_pred"}),
                how="inner", validate="one_to_one")
            pairwise.append({"task": task, "candidate": model,
                             "baseline": MODEL_2D, "n_paired_known_assay_observations": len(paired),
                             "delta_candidate_minus_2d_rmse": float(np.sqrt(np.mean((p3-y)**2))
                                                                         - np.sqrt(np.mean((p2-y)**2))),
                             "delta_all_assay_records_rmse": float(
                                 np.sqrt(np.mean((all_paired.candidate_pred-all_paired.y_true)**2))
                                 - np.sqrt(np.mean((all_paired.y_pred-all_paired.y_true)**2))),
                             "scaffold_bootstrap_ci95_low": float(lo),
                             "scaffold_bootstrap_ci95_high": float(hi),
                             "bootstrap_probability_candidate_better": float(np.mean(np.asarray(delta) < 0)),
                             "scaffold_groups": int(len(scaffolds))})

    pd.DataFrame(summary).to_csv(out_dir / "pooled_assay_metrics.csv", index=False)
    pd.DataFrame(pairwise).to_csv(out_dir / "paired_scaffold_bootstrap.csv", index=False)
    preds.to_csv(pred_path, index=False)
    pd.DataFrame(candidate_rows).to_csv(cand_path, index=False)
    pd.DataFrame(selection_rows).to_csv(select_path, index=False)
    metadata = {
        "status": "smoke_only" if args.max_outer_folds else "complete",
        "source_file": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
        "selection": "exact IC50, standard_relation='=', assay ID preserved, assay-molecule duplicates averaged",
        "minimum_assay_unique_molecules": args.min_assay_molecules,
        "assay_mean_handling": "subtract fit-only assay mean; predict using fit-only assay mean; global training mean fallback for unseen assay",
        "inner_model_selection_metric": "RMSE on inner-validation observations whose assay ID also appears in that inner-fit partition",
        "validation": "paired five-fold Bemis-Murcko GroupKFold from the supplied fold manifest; nested inner scaffold GroupKFold",
        "outer_scaffold_overlap": 0, "tasks": tasks,
        "outer_fold_manifest": str(fold_path.relative_to(ROOT) if fold_path.is_relative_to(ROOT) else fold_path),
        "outer_fold_manifest_sha256": sha256(fold_path),
        "features_2d_sha256": sha256(FEAT2), "features_3d_sha256": sha256(feature_path),
        "features_3d_file": str(feature_path.relative_to(ROOT) if feature_path.is_relative_to(ROOT) else feature_path),
        "features_3d_key": feature_key,
        "modeling_dataset_sha256": sha256(DATA),
        "warnings": ["Internal ChEMBL cross-validation, not external/prospective validation.",
                     "Assay-aware estimates apply to assays with at least the configured number of measured molecules.",
                     "The pre-specified scaffold partition was originally built on the aggregated molecular cohort; assay observations were grouped by the same scaffold assignments."],
        "assay_counts": assay_counts.groupby("task").agg(
            assays_total=("assay_chembl_id", "size"),
            assays_kept=("keep", "sum")).reset_index().to_dict("records"),
    }
    (out_dir / "audit.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(pd.DataFrame(pairwise).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
