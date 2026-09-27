#!/usr/bin/env python3
"""Assay-aware nested scaffold benchmark for 2D Tanimoto, 3D RBF, and MKL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial.distance import cdist
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import assay_context_3d_benchmark as base


ROOT = base.ROOT
SOURCE = base.SOURCE
FEAT2 = base.FEAT2
FOLDS = base.FOLDS
MODEL_2D = "assay_centered_2d_best"
MODEL_3D = "assay_centered_3d_rbf"
MODEL_MKL = "assay_context_mkl"
MODEL_CONTEXT = "assay_mean_only"
MODELS = {MODEL_2D, MODEL_3D, MODEL_MKL}


def molecular_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def tanimoto_kernel(features_2d: np.ndarray) -> np.ndarray:
    bits = features_2d[:, :2048]
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("First 2048 2D feature columns are expected to be binary Morgan bits")
    # Jaccard distance on binary vectors is exactly 1 - Tanimoto similarity.
    sim = 1.0 - cdist(bits.astype(bool), bits.astype(bool), metric="jaccard")
    sim = np.nan_to_num(sim, nan=1.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


def kernel_candidates(n_estimators: int, seed: int):
    families = {
        MODEL_2D: [(f"vector_{name}", "vector2", prototype)
                   for name, prototype in base.candidates("2d", seed, n_estimators)],
        MODEL_3D: [],
        MODEL_MKL: [],
    }
    for alpha in (0.01, 0.1, 1.0, 10.0):
        families[MODEL_2D].append((f"tanimoto_krr_alpha{alpha:g}", "tanimoto", alpha))
    for gamma_mult in (0.25, 1.0, 4.0):
        for alpha in (0.1, 1.0, 10.0):
            families[MODEL_3D].append((f"rbf3d_g{gamma_mult:g}_a{alpha:g}", "rbf3d", (gamma_mult, alpha)))
            for weight_3d in (0.25, 0.5, 0.75):
                families[MODEL_MKL].append((
                    f"mkl_w3d{weight_3d:g}_g{gamma_mult:g}_a{alpha:g}",
                    "mkl", (weight_3d, gamma_mult, alpha)))
    return families


def precompute_split_kernels(x3: np.ndarray, row_molecule_ids: np.ndarray,
                             fit_idx: np.ndarray, eval_idx: np.ndarray,
                             k2_molecule: np.ndarray):
    fit_mols = row_molecule_ids[fit_idx]
    eval_mols = row_molecule_ids[eval_idx]
    k2_fit = k2_molecule[np.ix_(fit_mols, fit_mols)]
    k2_eval = k2_molecule[np.ix_(eval_mols, fit_mols)]
    unique_fit = np.unique(fit_mols)
    all_mols = np.unique(np.concatenate([fit_mols, eval_mols]))
    scaler = StandardScaler()
    scaler.fit(x3[unique_fit])
    z = scaler.transform(x3[all_mols]).astype(np.float64)
    if not np.isfinite(z).all():
        raise FloatingPointError("Non-finite fold-standardized 3D features")
    local = {int(mol): j for j, mol in enumerate(all_mols)}
    fit_pos = np.asarray([local[int(mol)] for mol in fit_mols], dtype=np.int64)
    eval_pos = np.asarray([local[int(mol)] for mol in eval_mols], dtype=np.int64)
    sq = cdist(z, z, metric="sqeuclidean")
    if not np.isfinite(sq).all():
        raise FloatingPointError("Non-finite fold-local 3D squared distances")
    k3 = {}
    for gamma_mult in (0.25, 1.0, 4.0):
        gamma = float(gamma_mult) / max(1, x3.shape[1])
        full = np.exp(-gamma * sq).astype(np.float64)
        k3[gamma_mult] = (full[np.ix_(fit_pos, fit_pos)], full[np.ix_(eval_pos, fit_pos)])
    return {"tanimoto": (k2_fit, k2_eval), "rbf3d": k3}


def fit_assay_kernel_predict(kind: str, params, x: np.ndarray, frame: pd.DataFrame,
                             fit_idx: np.ndarray, eval_idx: np.ndarray,
                             split_kernels: dict, seed: int):
    if kind in {"vector2", "vector3", "vectorfused"}:
        return base.assay_centered_fit_predict(
            params, x, frame, fit_idx, eval_idx, seed)
    train = frame.iloc[fit_idx]
    pred = frame.iloc[eval_idx]
    assay_means = train.groupby("assay_chembl_id").y.mean()
    global_mean = float(train.y.mean())
    y_centered = train.y.to_numpy(dtype=np.float64) - train.assay_chembl_id.map(assay_means).to_numpy(dtype=np.float64)
    if kind == "tanimoto":
        k_train, k_eval = split_kernels["tanimoto"]
    elif kind == "rbf3d":
        k_train, k_eval = split_kernels["rbf3d"][float(params[0])]
    else:
        weight_3d, gamma_mult = float(params[0]), float(params[1])
        k2_train, k2_eval = split_kernels["tanimoto"]
        k3_train, k3_eval = split_kernels["rbf3d"][gamma_mult]
        k_train = (1.0 - weight_3d) * k2_train + weight_3d * k3_train
        k_eval = (1.0 - weight_3d) * k2_eval + weight_3d * k3_eval
    alpha = params if kind == "tanimoto" else float(params[-1])
    model = KernelRidge(alpha=float(alpha), kernel="precomputed")
    model.fit(k_train, y_centered)
    residual = model.predict(k_eval).astype(np.float64)
    assay_mean = pred.assay_chembl_id.map(assay_means).to_numpy(dtype=np.float64)
    seen = np.isfinite(assay_mean)
    assay_mean[~seen] = global_mean
    return assay_mean + residual, seen, assay_mean


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict:
    return base.metric_row(y, pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-assay-molecules", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--features-3d", default="results/unimol2_plus_rdkit_rich_3d_features.npz")
    parser.add_argument("--max-outer-folds", type=int, default=0, help="Debug only; output audit is smoke_only")
    parser.add_argument("--folds-file", default=str(FOLDS.relative_to(ROOT)))
    parser.add_argument("--output-dir", default="experiments/optimization_v1/assay_context_kernel_cv")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
    raw = raw.loc[(raw.standard_type.astype(str).str.upper() == "IC50")
                  & (raw.standard_relation.astype(str) == "=")
                  & raw.pchembl_value_final.notna()].copy()
    raw["task"] = raw.cell_line.map(base.TASKS)
    raw = raw.dropna(subset=["task", "assay_chembl_id", "canonical_smiles"]).copy()
    raw["smiles"] = raw.canonical_smiles.astype(str)
    raw["y"] = pd.to_numeric(raw.pchembl_value_final, errors="coerce")
    raw = raw.dropna(subset=["y"])

    f2 = np.load(FEAT2, allow_pickle=False)
    feature_path = Path(args.features_3d)
    if not feature_path.is_absolute():
        feature_path = ROOT / feature_path
    f3 = np.load(feature_path, allow_pickle=False)
    map2 = {str(smiles): i for i, smiles in enumerate(f2["smiles"])}
    map3 = {str(smiles): i for i, smiles in enumerate(f3["smiles"])}
    if set(map2) != set(map3):
        raise RuntimeError("2D/3D feature coverage differs")
    raw = raw.loc[raw.smiles.map(lambda smiles: smiles in map2)].copy()
    counts = raw.groupby(["task", "assay_chembl_id"]).smiles.nunique().rename("assay_n").reset_index()
    kept = counts.loc[counts.assay_n >= args.min_assay_molecules, ["task", "assay_chembl_id"]]
    raw = raw.merge(kept, on=["task", "assay_chembl_id"], how="inner", validate="many_to_one")
    raw = raw.reset_index(drop=True)
    raw["record_id"] = np.arange(len(raw), dtype=np.int64)

    molecules = raw.smiles.drop_duplicates().astype(str).tolist()
    row_map = {smiles: i for i, smiles in enumerate(molecules)}
    row_molecule_ids = raw.smiles.map(row_map).to_numpy(dtype=np.int64)
    x2 = f2["features"][np.asarray([map2[smiles] for smiles in molecules])].astype(np.float32)
    x3 = f3["features"][np.asarray([map3[smiles] for smiles in molecules])].astype(np.float32)
    if not np.isfinite(x2).all() or not np.isfinite(x3).all():
        raise RuntimeError("Non-finite feature matrix")
    k2 = tanimoto_kernel(x2)

    folds_path = Path(args.folds_file)
    if not folds_path.is_absolute():
        folds_path = ROOT / folds_path
    fold_table = pd.read_csv(folds_path)
    fold_maps = {task: dict(zip(part.smiles.astype(str), part.outer_fold.astype(int)))
                 for task, part in fold_table.groupby("task")}
    raw["molecule_id"] = row_molecule_ids
    raw["scaffold"] = raw.smiles.map(molecular_scaffold)
    raw["outer_fold"] = [fold_maps[t].get(s, -1) for t, s in zip(raw.task, raw.smiles)]
    if (raw.outer_fold < 0).any():
        raise RuntimeError("One or more assay records lack frozen scaffold assignments")

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
            if set(part.model.astype(str)) == MODELS:
                completed.add((str(task), int(fold)))

    for task in base.TASKS.values():
        part = raw.loc[raw.task == task].reset_index(drop=True)
        row_folds = part.outer_fold.to_numpy(dtype=int)
        groups = part.scaffold.astype(str).to_numpy()
        fold_ids = sorted(np.unique(row_folds))
        if args.max_outer_folds:
            fold_ids = fold_ids[:args.max_outer_folds]
        for outer_fold in fold_ids:
            if (task, int(outer_fold)) in completed:
                continue
            tr = np.flatnonzero(row_folds != outer_fold)
            te = np.flatnonzero(row_folds == outer_fold)
            if set(groups[tr]) & set(groups[te]):
                raise RuntimeError(f"Outer scaffold leakage in {task}/{outer_fold}")
            inner = GroupKFold(n_splits=args.inner_folds)
            inner_splits = list(inner.split(tr, part.y.to_numpy()[tr], groups[tr]))
            ids = part.molecule_id.to_numpy(dtype=np.int64)
            x2_task = x2[ids]
            inner_kernel_sets = []
            for fit_rel, val_rel in inner_splits:
                inner_kernel_sets.append(precompute_split_kernels(
                    x3, ids, tr[fit_rel], tr[val_rel], k2))
            outer_kernel_set = precompute_split_kernels(x3, ids, tr, te, k2)
            families = kernel_candidates(args.n_estimators, 20260925 + int(outer_fold))
            fit_means = part.iloc[tr].groupby("assay_chembl_id").y.mean()
            global_mean = float(part.iloc[tr].y.mean())
            context_test = part.iloc[te].assay_chembl_id.map(fit_means)
            seen_context = context_test.notna().to_numpy()
            context_pred = context_test.fillna(global_mean).to_numpy(dtype=float)

            for model_label, model_candidates in families.items():
                scored = []
                for name, kind, params in model_candidates:
                    oof = np.full(len(tr), np.nan, dtype=np.float64)
                    oof_known = np.zeros(len(tr), dtype=bool)
                    for inner_i, (fit_rel, val_rel) in enumerate(inner_splits):
                        fit_idx, val_idx = tr[fit_rel], tr[val_rel]
                        pred, seen, _ = fit_assay_kernel_predict(
                            kind, params, x2_task, part, fit_idx, val_idx,
                            inner_kernel_sets[inner_i], 20260925 + int(outer_fold) * 100 + inner_i)
                        oof[val_rel] = pred
                        oof_known[val_rel] = seen
                    if not np.isfinite(oof).all() or not oof_known.any():
                        raise RuntimeError(f"Incomplete inner OOF for {task}/{outer_fold}/{name}")
                    score = float(np.sqrt(mean_squared_error(
                        part.iloc[tr].y.to_numpy()[oof_known], oof[oof_known])))
                    candidate_rows.append({"task": task, "outer_fold": int(outer_fold),
                                           "model": model_label, "candidate": name,
                                           "inner_oof_rmse": score})
                    scored.append((score, name, kind, params))
                    print(f"{task} outer={outer_fold + 1}/5 {model_label} {name} inner_rmse={score:.4f}", flush=True)
                score, name, kind, params = min(scored, key=lambda row: row[0])
                pred, seen, assay_mean = fit_assay_kernel_predict(
                    kind, params, x2_task, part, tr, te,
                    outer_kernel_set, 20260925 + int(outer_fold))
                if not np.array_equal(seen, seen_context):
                    raise RuntimeError("Assay-support mismatch between fitted model and context baseline")
                selection_rows.append({"task": task, "outer_fold": int(outer_fold), "model": model_label,
                                       "selected_candidate": name, "inner_oof_rmse": score})
                for j, idx in enumerate(te):
                    row = part.iloc[idx]
                    pred_rows.append({"record_id": int(row.record_id), "task": task,
                                      "outer_fold": int(outer_fold), "assay_chembl_id": str(row.assay_chembl_id),
                                      "smiles": str(row.smiles), "scaffold": str(row.scaffold),
                                      "y_true": float(row.y), "assay_train_n": int((part.iloc[tr].assay_chembl_id == row.assay_chembl_id).sum()),
                                      "assay_seen_in_training": bool(seen[j]), "model": model_label,
                                      "y_pred": float(pred[j])})

            for j, idx in enumerate(te):
                row = part.iloc[idx]
                pred_rows.append({"record_id": int(row.record_id), "task": task,
                                  "outer_fold": int(outer_fold), "assay_chembl_id": str(row.assay_chembl_id),
                                  "smiles": str(row.smiles), "scaffold": str(row.scaffold),
                                  "y_true": float(row.y), "assay_train_n": int((part.iloc[tr].assay_chembl_id == row.assay_chembl_id).sum()),
                                  "assay_seen_in_training": bool(seen_context[j]), "model": MODEL_CONTEXT,
                                  "y_pred": float(context_pred[j])})
            pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
            pd.DataFrame(candidate_rows).to_csv(cand_path, index=False)
            pd.DataFrame(selection_rows).to_csv(select_path, index=False)
            completed.add((task, int(outer_fold)))

    preds = pd.DataFrame(pred_rows)
    summaries = []
    pairwise = []
    rng = np.random.default_rng(20260925)
    for task, task_preds in preds.groupby("task"):
        by_model = {model: group.set_index("record_id") for model, group in task_preds.groupby("model")}
        for model in (MODEL_2D, MODEL_3D, MODEL_MKL, MODEL_CONTEXT):
            part = task_preds.loc[task_preds.model == model]
            known = part.loc[part.assay_seen_in_training]
            unknown = part.loc[~part.assay_seen_in_training]
            summaries.append({"task": task, "model": model, **metric_row(part.y_true.to_numpy(), part.y_pred.to_numpy()),
                              "known_assay_coverage": float(part.assay_seen_in_training.mean()),
                              "known_assay_rmse": metric_row(known.y_true.to_numpy(), known.y_pred.to_numpy())["rmse"],
                              "unseen_assay_rmse": metric_row(unknown.y_true.to_numpy(), unknown.y_pred.to_numpy())["rmse"] if len(unknown) else float("nan"),
                              "n_assays_tested": int(part.assay_chembl_id.nunique()),
                              "n_unique_molecules": int(part.smiles.nunique())})
        baseline = by_model[MODEL_2D]
        for candidate in (MODEL_3D, MODEL_MKL, MODEL_CONTEXT):
            comp = by_model[candidate]
            paired = baseline.loc[baseline.assay_seen_in_training, ["scaffold", "y_true", "y_pred"]].join(
                comp.loc[comp.assay_seen_in_training, ["y_true", "y_pred"]].rename(
                    columns={"y_true": "candidate_y", "y_pred": "candidate_pred"}),
                how="inner", validate="one_to_one")
            if len(paired) != int(baseline.assay_seen_in_training.sum()) or not np.allclose(
                    paired.y_true.to_numpy(), paired.candidate_y.to_numpy(), atol=1e-7, rtol=0):
                raise RuntimeError(f"Could not pair exact known-assay records for {task}/{candidate}")
            y, p2, p3 = paired.y_true.to_numpy(), paired.y_pred.to_numpy(), paired.candidate_pred.to_numpy()
            scaffolds = paired.scaffold.unique()
            by_scaffold = {s: np.flatnonzero(paired.scaffold.to_numpy() == s) for s in scaffolds}
            deltas = []
            for _ in range(2000):
                sampled = rng.choice(scaffolds, size=len(scaffolds), replace=True)
                idx = np.concatenate([by_scaffold[s] for s in sampled])
                deltas.append(float(np.sqrt(np.mean((p3[idx] - y[idx]) ** 2))
                                       - np.sqrt(np.mean((p2[idx] - y[idx]) ** 2))))
            lo, hi = np.quantile(deltas, [0.025, 0.975])
            pairwise.append({"task": task, "candidate": candidate, "baseline": MODEL_2D,
                             "n_paired_known_assay_observations": int(len(paired)),
                             "delta_candidate_minus_2d_rmse": float(np.sqrt(np.mean((p3-y)**2)) - np.sqrt(np.mean((p2-y)**2))),
                             "scaffold_bootstrap_ci95_low": float(lo),
                             "scaffold_bootstrap_ci95_high": float(hi),
                             "bootstrap_probability_candidate_better": float(np.mean(np.asarray(deltas) < 0)),
                             "scaffold_groups": int(len(scaffolds))})

    pd.DataFrame(summaries).to_csv(out_dir / "pooled_assay_metrics.csv", index=False)
    pd.DataFrame(pairwise).to_csv(out_dir / "paired_scaffold_bootstrap.csv", index=False)
    preds.to_csv(pred_path, index=False)
    pd.DataFrame(candidate_rows).to_csv(cand_path, index=False)
    pd.DataFrame(selection_rows).to_csv(select_path, index=False)
    manifest_path = Path(args.folds_file)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    audit = {
        "status": "smoke_only" if args.max_outer_folds else "complete",
        "source_file": str(SOURCE.relative_to(ROOT)), "source_sha256": base.sha256(SOURCE),
        "selection": "exact IC50; assay ID retained; assays with >=10 distinct molecules; fit-only assay means",
        "primary_metric": "known-assay RMSE",
        "validation": "nested scaffold GroupKFold; same frozen outer folds for all model families",
        "outer_scaffold_overlap": 0,
        "outer_fold_manifest": str(manifest_path.relative_to(ROOT) if manifest_path.is_relative_to(ROOT) else manifest_path),
        "outer_fold_manifest_sha256": base.sha256(manifest_path),
        "features_2d_sha256": base.sha256(FEAT2),
        "features_3d_file": str(feature_path.relative_to(ROOT) if feature_path.is_relative_to(ROOT) else feature_path),
        "features_3d_sha256": base.sha256(feature_path),
        "assay_records": {str(k): int(v) for k, v in raw.groupby("task").size().items()},
        "assay_counts": {str(k): int(v) for k, v in raw.groupby("task").assay_chembl_id.nunique().items()},
        "warnings": ["Internal ChEMBL cross-validation only; no prospective external validation.",
                     "Stage 3 was specified after earlier exploratory outer-split runs; any apparent improvement requires confirmation on the two pre-generated independent scaffold manifests."],
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(pairwise).to_string(index=False))


if __name__ == "__main__":
    main()
