#!/usr/bin/env python3
"""Nested shared-scaffold evaluation of task-conditioned 2D and 2D+Uni-Mol2 models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import assay_context_3d_benchmark as base

ROOT = base.ROOT
SOURCE = base.SOURCE
FEAT2 = base.FEAT2
FEAT3 = ROOT / "results/unimol2_84m_10conf_pooled_cls.npz"
TASKS = ("hepg2_pIC50", "hct116_pIC50")
MODEL_2D = "joint_task_conditioned_2d"
MODEL_3D = "joint_task_conditioned_2d_plus_unimol2_3d"
MODEL_CONTEXT = "assay_mean_only"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_map(path: Path, key: str) -> tuple[dict[str, int], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    smiles = data["smiles"].astype(str)
    matrix = data[key].astype(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"Non-finite feature matrix: {path}")
    return {s: i for i, s in enumerate(smiles)}, matrix


def build_pca_views(x: np.ndarray, unique_fit_ids: np.ndarray,
                    all_molecule_ids: np.ndarray, ranks: tuple[int, ...], seed: int):
    """Fit scaling/PCA on unique training molecules only; transform requested rows."""
    views: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for rank in ranks:
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x[unique_fit_ids].astype(np.float64))
        pca = PCA(n_components=rank, whiten=True, svd_solver="randomized",
                  iterated_power=2, random_state=seed + rank)
        pca.fit(x_fit)
        all_scaled = scaler.transform(x[all_molecule_ids].astype(np.float64))
        z = pca.transform(all_scaled).astype(np.float32)
        if not np.isfinite(z).all():
            raise FloatingPointError("Non-finite fold-local PCA scores")
        views[rank] = (z, pca.explained_variance_ratio_.astype(float))
    return views


def design(z: np.ndarray, task_code: np.ndarray) -> np.ndarray:
    """Shared chemistry effect plus cell-line-specific deviations (partial pooling)."""
    onehot = np.column_stack([task_code == 0, task_code == 1]).astype(np.float32)
    specific = np.concatenate([z * onehot[:, 0:1], z * onehot[:, 1:2]], axis=1)
    return np.concatenate([z, onehot, specific], axis=1).astype(np.float32)


def fit_means(frame: pd.DataFrame, idx: np.ndarray):
    train = frame.iloc[idx]
    means = train.groupby(["task", "assay_chembl_id"]).y.mean()
    global_means = train.groupby("task").y.mean().to_dict()
    key = list(zip(train.task.astype(str), train.assay_chembl_id.astype(str)))
    y_centered = train.y.to_numpy(dtype=np.float64) - means.reindex(
        pd.MultiIndex.from_tuples(key, names=["task", "assay_chembl_id"])
    ).to_numpy(dtype=np.float64)
    return means, global_means, y_centered


def restore_means(frame: pd.DataFrame, idx: np.ndarray, means, global_means):
    rows = frame.iloc[idx]
    keys = pd.MultiIndex.from_arrays(
        [rows.task.astype(str).to_numpy(), rows.assay_chembl_id.astype(str).to_numpy()],
        names=["task", "assay_chembl_id"])
    assay_mean = means.reindex(keys).to_numpy(dtype=np.float64)
    seen = np.isfinite(assay_mean)
    fallback = rows.task.map(global_means).to_numpy(dtype=np.float64)
    assay_mean[~seen] = fallback[~seen]
    return assay_mean, seen


def candidate_grid(family: str):
    rows = []
    if family == MODEL_2D:
        for p2 in (32, 64):
            for alpha in (1.0, 10.0, 100.0):
                rows.append({"p2": p2, "p3": 0, "w3": 0.0, "alpha": alpha})
    elif family == MODEL_3D:
        for p2 in (32, 64):
            for p3 in (16, 32):
                for w3 in (0.25, 0.5, 0.75):
                    for alpha in (1.0, 10.0, 100.0):
                        rows.append({"p2": p2, "p3": p3, "w3": w3, "alpha": alpha})
    else:
        raise ValueError(family)
    return rows


def candidate_name(family: str, row: dict) -> str:
    if family == MODEL_2D:
        return f"pca2d{row['p2']}_partial_pool_ridge_a{row['alpha']:g}"
    return (f"pca2d{row['p2']}_unimol2d3d{row['p3']}_w{row['w3']:g}_"
            f"partial_pool_ridge_a{row['alpha']:g}")


def task_rmse(frame: pd.DataFrame, idx: np.ndarray, pred: np.ndarray,
              known: np.ndarray) -> dict[str, float]:
    out = {}
    tasks = frame.task.iloc[idx].astype(str).to_numpy()
    y = frame.y.iloc[idx].to_numpy(dtype=np.float64)
    for task in TASKS:
        use = known & (tasks == task)
        out[task] = float(np.sqrt(mean_squared_error(y[use], pred[use]))) if use.any() else float("nan")
    return out


def score_candidate(frame: pd.DataFrame, fit_idx: np.ndarray, eval_idx: np.ndarray,
                    z2_fit: np.ndarray, z2_eval: np.ndarray,
                    z3_fit: np.ndarray | None, z3_eval: np.ndarray | None,
                    task_code: np.ndarray, config: dict):
    if z3_fit is None or z3_eval is None:
        z_fit, z = z2_fit, z2_eval
    else:
        z_fit = np.concatenate([z2_fit, config["w3"] * z3_fit], axis=1)
        z = np.concatenate([z2_eval, config["w3"] * z3_eval], axis=1)
    x_fit = design(z_fit, task_code[fit_idx])
    x_eval = design(z, task_code[eval_idx])
    means, global_means, y_centered = fit_means(frame, fit_idx)
    model = Ridge(alpha=float(config["alpha"]), solver="lsqr")
    with threadpool_limits(limits=1, user_api="blas"):
        model.fit(x_fit, y_centered)
        residual = model.predict(x_eval).astype(np.float64)
    restored, seen = restore_means(frame, eval_idx, means, global_means)
    return restored + residual, seen


def paired_bootstrap(preds: pd.DataFrame, task: str, candidate: str,
                     baseline: str, seed: int, n_boot: int = 2000) -> dict:
    p = preds.loc[(preds.task == task) & (preds.model == baseline) & preds.known_assay].set_index("record_id")
    c = preds.loc[(preds.task == task) & (preds.model == candidate) & preds.known_assay].set_index("record_id")
    joined = p[["scaffold", "y_true", "y_pred"]].join(
        c[["y_true", "y_pred"]].rename(columns={"y_true": "y_candidate", "y_pred": "p_candidate"}),
        how="inner", validate="one_to_one")
    if len(joined) != len(p) or not np.allclose(joined.y_true, joined.y_candidate, atol=1e-7, rtol=0):
        raise RuntimeError(f"Unpaired exact OOF rows for {task}: {candidate} vs {baseline}")
    y = joined.y_true.to_numpy(dtype=np.float64)
    p0 = joined.y_pred.to_numpy(dtype=np.float64)
    p1 = joined.p_candidate.to_numpy(dtype=np.float64)
    groups = joined.scaffold.astype(str).to_numpy()
    unique = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([by_group[g] for g in chosen])
        deltas[b] = np.sqrt(np.mean((p1[ix] - y[ix]) ** 2)) - np.sqrt(np.mean((p0[ix] - y[ix]) ** 2))
    return {
        "task": task, "candidate": candidate, "baseline": baseline,
        "n_paired_known_assay_observations": int(len(joined)),
        "delta_candidate_minus_baseline_rmse": float(np.sqrt(np.mean((p1-y)**2)) - np.sqrt(np.mean((p0-y)**2))),
        "scaffold_bootstrap_ci95_low": float(np.quantile(deltas, 0.025)),
        "scaffold_bootstrap_ci95_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_probability_candidate_better": float(np.mean(deltas < 0)),
        "scaffold_groups": int(len(unique)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--max-outer-folds", type=int, default=0,
                        help="Development only; marks all output smoke_only")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.folds_file)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
    raw = raw.loc[(raw.standard_type.astype(str).str.upper() == "IC50")
                  & (raw.standard_relation.astype(str) == "=")
                  & raw.pchembl_value_final.notna()].copy()
    raw["task"] = raw.cell_line.map(base.TASKS)
    raw = raw.dropna(subset=["task", "assay_chembl_id", "canonical_smiles"]).copy()
    raw["smiles"] = raw.canonical_smiles.astype(str)
    raw["y"] = pd.to_numeric(raw.pchembl_value_final, errors="coerce")
    raw = raw.dropna(subset=["y"])

    map2, full2 = feature_map(FEAT2, "features")
    map3, full3 = feature_map(FEAT3, "cls_repr")
    if set(map2) != set(map3):
        raise RuntimeError("2D and Uni-Mol2 molecule keys differ")
    raw = raw.loc[raw.smiles.isin(map2)].copy()
    assay_counts = raw.groupby(["task", "assay_chembl_id"]).smiles.nunique().rename("n").reset_index()
    kept = assay_counts.loc[assay_counts.n >= 10, ["task", "assay_chembl_id"]]
    raw = raw.merge(kept, on=["task", "assay_chembl_id"], how="inner", validate="many_to_one")
    manifest = pd.read_csv(manifest_path)
    if manifest.groupby("scaffold").outer_fold.nunique().max() != 1:
        raise RuntimeError("Manifest does not have a global scaffold fold assignment")
    fold_by_smiles = manifest.drop_duplicates("smiles").set_index("smiles").outer_fold.to_dict()
    scaffold_by_smiles = manifest.drop_duplicates("smiles").set_index("smiles").scaffold.to_dict()
    raw = raw.loc[raw.smiles.isin(fold_by_smiles)].copy().reset_index(drop=True)
    raw["outer_fold"] = raw.smiles.map(fold_by_smiles).astype(int)
    raw["scaffold"] = raw.smiles.map(scaffold_by_smiles).astype(str)
    raw["record_id"] = np.arange(len(raw), dtype=np.int64)
    if raw.groupby("smiles").outer_fold.nunique().max() != 1:
        raise RuntimeError("Molecule appears in multiple outer folds")
    if raw.groupby("scaffold").outer_fold.nunique().max() != 1:
        raise RuntimeError("Scaffold appears in multiple outer folds across tasks")
    molecules = raw.smiles.drop_duplicates().astype(str).tolist()
    mol_index = {s: i for i, s in enumerate(molecules)}
    raw["molecule_id"] = raw.smiles.map(mol_index).astype(int)
    x2 = full2[np.asarray([map2[s] for s in molecules])]
    x3 = full3[np.asarray([map3[s] for s in molecules])]
    task_code_map = {task: i for i, task in enumerate(TASKS)}
    task_code = raw.task.map(task_code_map).to_numpy(dtype=np.int64)

    preds_path = out / "outer_oof_predictions.csv"
    scores_path = out / "inner_candidate_scores.csv"
    selects_path = out / "inner_model_selections.csv"
    predictions: list[dict] = []
    candidate_rows: list[dict] = []
    selection_rows: list[dict] = []
    completed_folds: set[int] = set()
    if preds_path.exists() and selects_path.exists():
        prior = pd.read_csv(selects_path)
        completed_folds = set(prior.outer_fold.astype(int).unique())
        predictions = pd.read_csv(preds_path).to_dict("records")
        candidate_rows = pd.read_csv(scores_path).to_dict("records") if scores_path.exists() else []
        selection_rows = prior.to_dict("records")

    outer_folds = sorted(raw.outer_fold.unique().tolist())
    if args.max_outer_folds:
        outer_folds = outer_folds[:args.max_outer_folds]
    for outer_fold in outer_folds:
        if int(outer_fold) in completed_folds:
            continue
        tr = np.flatnonzero(raw.outer_fold.to_numpy() != outer_fold)
        te = np.flatnonzero(raw.outer_fold.to_numpy() == outer_fold)
        if set(raw.scaffold.iloc[tr]) & set(raw.scaffold.iloc[te]):
            raise RuntimeError("Outer scaffold leakage")
        outer_inner = GroupKFold(n_splits=args.inner_folds)
        inner_splits = list(outer_inner.split(tr, raw.y.to_numpy()[tr], raw.scaffold.to_numpy()[tr]))

        # Each inner split gets training-only PCA transforms. The held-out inner
        # molecules are transformed, never used to fit either scaler or PCA.
        cache = []
        for inner_i, (fit_rel, val_rel) in enumerate(inner_splits):
            fit_idx, val_idx = tr[fit_rel], tr[val_rel]
            fit_molecules = np.unique(raw.molecule_id.to_numpy()[fit_idx])
            eval_molecules = np.unique(raw.molecule_id.to_numpy()[val_idx])
            requested = np.unique(np.concatenate([fit_molecules, eval_molecules]))
            p2 = build_pca_views(x2, fit_molecules, requested, (32, 64),
                                 seed=20261100 + int(outer_fold)*100 + inner_i)
            p3 = build_pca_views(x3, fit_molecules, requested, (16, 32),
                                 seed=20261200 + int(outer_fold)*100 + inner_i)
            position = {int(mol): i for i, mol in enumerate(requested)}
            fit_pos = np.asarray([position[int(m)] for m in raw.molecule_id.to_numpy()[fit_idx]])
            val_pos = np.asarray([position[int(m)] for m in raw.molecule_id.to_numpy()[val_idx]])
            row_views = {}
            for rank, (view, _) in p2.items():
                row_views[("2d", rank, "fit")] = view[fit_pos]
                row_views[("2d", rank, "eval")] = view[val_pos]
            for rank, (view, _) in p3.items():
                row_views[("3d", rank, "fit")] = view[fit_pos]
                row_views[("3d", rank, "eval")] = view[val_pos]
            cache.append((fit_idx, val_idx, row_views))

        chosen_by_family = {}
        for family in (MODEL_2D, MODEL_3D):
            scored = []
            for config in candidate_grid(family):
                oof = np.full(len(tr), np.nan, dtype=np.float64)
                known = np.zeros(len(tr), dtype=bool)
                for inner_i, (fit_idx, val_idx, row_views) in enumerate(cache):
                    z2_fit = row_views[("2d", config["p2"], "fit")]
                    z2_eval = row_views[("2d", config["p2"], "eval")]
                    z3_fit = None if family == MODEL_2D else row_views[("3d", config["p3"], "fit")]
                    z3_eval = None if family == MODEL_2D else row_views[("3d", config["p3"], "eval")]
                    pred, seen = score_candidate(
                        raw, fit_idx, val_idx, z2_fit, z2_eval, z3_fit, z3_eval,
                        task_code, config)
                    oof[np.searchsorted(tr, val_idx)] = pred
                    known[np.searchsorted(tr, val_idx)] = seen
                if not np.isfinite(oof).all():
                    raise RuntimeError(f"Incomplete inner OOF for {family}/{config}")
                task_scores = task_rmse(raw, tr, oof, known)
                finite_scores = [v for v in task_scores.values() if np.isfinite(v)]
                if len(finite_scores) != len(TASKS):
                    raise RuntimeError("Inner known-assay support is missing a task")
                score = float(np.mean(finite_scores))
                name = candidate_name(family, config)
                candidate_rows.append({"outer_fold": int(outer_fold), "family": family,
                                       "candidate": name, **config,
                                       "inner_equal_task_mean_known_assay_rmse": score,
                                       "inner_hepg2_known_assay_rmse": task_scores[TASKS[0]],
                                       "inner_hct116_known_assay_rmse": task_scores[TASKS[1]],
                                       "inner_known_assay_records": int(known.sum())})
                scored.append((score, name, config, oof, known))
                print(f"outer={int(outer_fold)+1}/5 {family} {name} "
                      f"inner_mean_rmse={score:.4f}", flush=True)
            best = min(scored, key=lambda r: r[0])
            chosen_by_family[family] = best

        # Refit the selected paired models on all outer-training molecules.
        fit_molecules = np.unique(raw.molecule_id.to_numpy()[tr])
        eval_molecules = np.unique(raw.molecule_id.to_numpy()[te])
        requested = np.unique(np.concatenate([fit_molecules, eval_molecules]))
        p2 = build_pca_views(x2, fit_molecules, requested, (32, 64), seed=20261300 + int(outer_fold))
        p3 = build_pca_views(x3, fit_molecules, requested, (16, 32), seed=20261400 + int(outer_fold))
        position = {int(mol): i for i, mol in enumerate(requested)}
        fit_pos = np.asarray([position[int(m)] for m in raw.molecule_id.to_numpy()[tr]])
        test_pos = np.asarray([position[int(m)] for m in raw.molecule_id.to_numpy()[te]])

        for family, model_name in ((MODEL_2D, MODEL_2D), (MODEL_3D, MODEL_3D)):
            _, selected_name, config, _, _ = chosen_by_family[family]
            z2 = p2[config["p2"]][0]
            z3 = None if family == MODEL_2D else p3[config["p3"]][0]
            # Build matrices by molecule-position lookup; raw IDs are global, PCA
            # view rows are ordered by the `requested` molecule IDs.
            all_view2 = z2
            fit_z2 = all_view2[fit_pos]
            eval_z2 = all_view2[test_pos]
            if z3 is not None:
                fit_z = np.concatenate([fit_z2, config["w3"] * z3[fit_pos]], axis=1)
                eval_z = np.concatenate([eval_z2, config["w3"] * z3[test_pos]], axis=1)
            else:
                fit_z, eval_z = fit_z2, eval_z2
            x_fit = design(fit_z, task_code[tr])
            x_eval = design(eval_z, task_code[te])
            means, global_means, y_centered = fit_means(raw, tr)
            model = Ridge(alpha=float(config["alpha"]), solver="lsqr")
            with threadpool_limits(limits=1, user_api="blas"):
                model.fit(x_fit, y_centered)
                residual = model.predict(x_eval).astype(np.float64)
            restored, seen = restore_means(raw, te, means, global_means)
            pred = restored + residual
            selection_rows.append({"outer_fold": int(outer_fold), "family": family,
                                   "selected_candidate": selected_name,
                                   "inner_equal_task_mean_known_assay_rmse": float(chosen_by_family[family][0]),
                                   **config})
            for j, idx in enumerate(te):
                row = raw.iloc[idx]
                predictions.append({"record_id": int(row.record_id), "task": str(row.task),
                                    "outer_fold": int(outer_fold), "assay_chembl_id": str(row.assay_chembl_id),
                                    "smiles": str(row.smiles), "scaffold": str(row.scaffold),
                                    "y_true": float(row.y), "assay_seen_in_training": bool(seen[j]),
                                    "known_assay": bool(seen[j]), "model": model_name,
                                    "y_pred": float(pred[j])})

        means, global_means, _ = fit_means(raw, tr)
        context_pred, context_seen = restore_means(raw, te, means, global_means)
        for j, idx in enumerate(te):
            row = raw.iloc[idx]
            predictions.append({"record_id": int(row.record_id), "task": str(row.task),
                                "outer_fold": int(outer_fold), "assay_chembl_id": str(row.assay_chembl_id),
                                "smiles": str(row.smiles), "scaffold": str(row.scaffold),
                                "y_true": float(row.y), "assay_seen_in_training": bool(context_seen[j]),
                                "known_assay": bool(context_seen[j]), "model": MODEL_CONTEXT,
                                "y_pred": float(context_pred[j])})
        pd.DataFrame(predictions).to_csv(preds_path, index=False)
        pd.DataFrame(candidate_rows).to_csv(scores_path, index=False)
        pd.DataFrame(selection_rows).to_csv(selects_path, index=False)
        completed_folds.add(int(outer_fold))
        print(f"completed outer fold {int(outer_fold)+1}/5", flush=True)

    pred_frame = pd.DataFrame(predictions)
    summaries = []
    for task, task_frame in pred_frame.groupby("task"):
        for model, group in task_frame.groupby("model"):
            known = group.loc[group.known_assay]
            summaries.append({"task": task, "model": model, "n": int(len(group)),
                              "known_assay_coverage": float(group.known_assay.mean()),
                              "known_assay_n": int(len(known)),
                              "known_assay_rmse": float(np.sqrt(mean_squared_error(known.y_true, known.y_pred))),
                              "known_assay_mae": float(mean_absolute_error(known.y_true, known.y_pred)),
                              "known_assay_r2": float(r2_score(known.y_true, known.y_pred)),
                              "n_unique_molecules": int(group.smiles.nunique()),
                              "n_scaffolds": int(group.scaffold.nunique())})
    bootstrap = []
    for task in TASKS:
        bootstrap.append(paired_bootstrap(pred_frame, task, MODEL_3D, MODEL_2D,
                                          seed=20261500 + (0 if task == TASKS[0] else 1)))
    pd.DataFrame(summaries).to_csv(out / "pooled_assay_metrics.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(out / "paired_scaffold_bootstrap.csv", index=False)
    pred_frame.to_csv(preds_path, index=False)
    manifest_meta = manifest_path.with_suffix(manifest_path.suffix + ".audit.json")
    audit = {
        "status": "smoke_only" if args.max_outer_folds else "complete",
        "source_sha256": sha256(SOURCE), "feature_2d_sha256": sha256(FEAT2),
        "feature_unimol2_sha256": sha256(FEAT3), "fold_manifest_sha256": sha256(manifest_path),
        "fold_manifest_audit_sha256": sha256(manifest_meta) if manifest_meta.exists() else None,
        "filter": "exact IC50 relation, non-null pChEMBL, feature-covered molecules, >=10 unique molecules per task-assay",
        "primary_metric": "pooled known-assay RMSE by cell-line task; assay means fit on training rows only",
        "outer_validation": "5-fold Bemis-Murcko scaffold split with one global scaffold-to-fold assignment shared across both cell lines",
        "inner_validation": f"{args.inner_folds}-fold GroupKFold by shared scaffold; candidate selection minimizes unweighted mean of task-specific known-assay RMSE",
        "model": "shared Ridge on train-fold StandardScaler+whitened PCA; common molecular effect plus cell-line-specific feature deviations; Uni-Mol2 is a nonzero weighted branch in every 3D-aware candidate",
        "pca_ranks": {"2d": [32, 64], "unimol2_3d": [16, 32]},
        "3d_weights": [0.25, 0.5, 0.75], "ridge_alphas": [1.0, 10.0, 100.0],
        "shared_scaffold_assignment_count": int(manifest.scaffold.nunique()),
        "cross_task_scaffold_leakage": 0, "outer_molecule_overlap": 0,
        "warnings": ["Internal ChEMBL scaffold CV; no prospective biological validation.",
                     "This is a 2D+Uni-Mol2 3D fusion result, not a pure 3D-only model.",
                     "Uni-Mol2 CLS is a frozen conformer-ensemble representation; PCA is fold-local."]
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(bootstrap).to_string(index=False))


if __name__ == "__main__":
    main()
