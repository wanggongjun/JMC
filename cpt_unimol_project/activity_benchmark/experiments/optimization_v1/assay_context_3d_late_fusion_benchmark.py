#!/usr/bin/env python3
"""Nested scaffold CV for shrinkage late fusion of matched 2D and Uni-Mol2 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import assay_context_3d_benchmark as assay_base
import assay_context_kernel_benchmark as kernel_base


ROOT = assay_base.ROOT
SOURCE = assay_base.SOURCE
FEAT2 = assay_base.FEAT2
MODEL_2D = "assay_centered_2d_reference"
MODEL_3D_AWARE = "assay_centered_3d_aware_strategy"
MODEL_CONTEXT = "assay_mean_only"
FAMILIES = {MODEL_2D, MODEL_3D_AWARE}


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def make_candidates(n_estimators: int, seed: int, n_2d: int, n_3d: int):
    families = {MODEL_2D: [], MODEL_3D_AWARE: []}
    for name, proto in assay_base.candidates("2d", seed, n_estimators):
        families[MODEL_2D].append((f"2d_vector_{name}", "vector2", proto, "2d"))
    for alpha in (0.01, 0.1, 1.0, 10.0):
        families[MODEL_2D].append((f"2d_tanimoto_krr_a{alpha:g}", "tanimoto", alpha, None))

    for n_components in (16, 32, 64):
        for c in (0.1, 1.0):
            pca_3d = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components, whiten=True, svd_solver="randomized",
                    iterated_power=2, random_state=seed),
            )
            svr = SVR(C=c, epsilon=0.15, kernel="rbf")
            families[MODEL_3D_AWARE].append((
                f"3d_pca{n_components}_svr_c{c:g}", "vector3",
                Pipeline([("projection", pca_3d), ("regressor", svr)]), "3d"))
    return families


def precompute_tanimoto_split(k2: np.ndarray, row_molecule_ids: np.ndarray,
                              fit_idx: np.ndarray, eval_idx: np.ndarray) -> dict:
    fit_mols = row_molecule_ids[fit_idx]
    eval_mols = row_molecule_ids[eval_idx]
    return {"tanimoto": (k2[np.ix_(fit_mols, fit_mols)],
                         k2[np.ix_(eval_mols, fit_mols)])}


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict:
    return assay_base.metric_row(y, pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-assay-molecules", type=int, default=10)
    parser.add_argument("--min-rotatable-bonds", type=int, default=0,
                        help="Optional predeclared flexible-molecule domain filter; 0 keeps the full cohort")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--features-3d", default="results/unimol2_plus_rdkit_rich_3d_features.npz")
    parser.add_argument("--max-outer-folds", type=int, default=0, help="Debug only; output audit is smoke_only")
    parser.add_argument("--split-level", choices=("scaffold", "molecule"), default="scaffold")
    parser.add_argument("--folds-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(SOURCE, encoding="utf-8-sig")
    raw = raw.loc[(raw.standard_type.astype(str).str.upper() == "IC50")
                  & (raw.standard_relation.astype(str) == "=")
                  & raw.pchembl_value_final.notna()].copy()
    raw["task"] = raw.cell_line.map(assay_base.TASKS)
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
        raise RuntimeError("2D and 3D feature key sets differ")
    raw = raw.loc[raw.smiles.map(lambda smiles: smiles in map2)].copy()
    if args.min_rotatable_bonds:
        rotatable_cache = {}
        def n_rotatable(smiles: str) -> int:
            if smiles not in rotatable_cache:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES for rotatable-bond filter: {smiles}")
                rotatable_cache[smiles] = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
            return rotatable_cache[smiles]
        raw["rotatable_bonds"] = raw.smiles.astype(str).map(n_rotatable)
        raw = raw.loc[raw.rotatable_bonds >= args.min_rotatable_bonds].copy()
        if raw.empty:
            raise RuntimeError("Rotatable-bond domain filter removed every source record")
    counts = raw.groupby(["task", "assay_chembl_id"]).smiles.nunique().rename("assay_n").reset_index()
    kept = counts.loc[counts.assay_n >= args.min_assay_molecules, ["task", "assay_chembl_id"]]
    raw = raw.merge(kept, on=["task", "assay_chembl_id"], how="inner", validate="many_to_one")
    raw = raw.reset_index(drop=True)
    raw["record_id"] = np.arange(len(raw), dtype=np.int64)

    molecules = raw.smiles.drop_duplicates().astype(str).tolist()
    molecule_map = {smiles: i for i, smiles in enumerate(molecules)}
    molecule_ids_all = raw.smiles.map(molecule_map).to_numpy(dtype=np.int64)
    x2 = f2["features"][np.asarray([map2[smiles] for smiles in molecules])].astype(np.float32)
    x3 = f3["features"][np.asarray([map3[smiles] for smiles in molecules])].astype(np.float32)
    if not np.isfinite(x2).all() or not np.isfinite(x3).all():
        raise RuntimeError("Non-finite feature arrays")
    k2 = kernel_base.tanimoto_kernel(x2)

    manifest_path = Path(args.folds_file)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    fold_table = pd.read_csv(manifest_path)
    fold_maps = {task: dict(zip(part.smiles.astype(str), part.outer_fold.astype(int)))
                 for task, part in fold_table.groupby("task")}
    raw["molecule_id"] = molecule_ids_all
    raw["scaffold"] = raw.smiles.map(scaffold)
    raw["outer_fold"] = [fold_maps[t].get(s, -1) for t, s in zip(raw.task, raw.smiles)]
    if (raw.outer_fold < 0).any():
        raise RuntimeError("Missing frozen scaffold-fold assignment")

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
            if set(part.model.astype(str)) == FAMILIES:
                completed.add((str(task), int(fold)))

    for task in assay_base.TASKS.values():
        part = raw.loc[raw.task == task].reset_index(drop=True)
        row_folds = part.outer_fold.to_numpy(dtype=int)
        groups = (part.scaffold if args.split_level == "scaffold" else part.smiles).astype(str).to_numpy()
        fold_ids = sorted(np.unique(row_folds))
        if args.max_outer_folds:
            fold_ids = fold_ids[:args.max_outer_folds]
        for outer_fold in fold_ids:
            if (task, int(outer_fold)) in completed:
                continue
            tr = np.flatnonzero(row_folds != outer_fold)
            te = np.flatnonzero(row_folds == outer_fold)
            if args.split_level == "scaffold":
                if set(groups[tr]) & set(groups[te]):
                    raise RuntimeError(f"Outer scaffold leakage in {task}/{outer_fold}")
            elif set(part.smiles.iloc[tr]) & set(part.smiles.iloc[te]):
                raise RuntimeError(f"Outer molecule leakage in {task}/{outer_fold}")
            inner = GroupKFold(n_splits=args.inner_folds)
            inner_splits = list(inner.split(tr, part.y.to_numpy()[tr], groups[tr]))
            molecule_ids = part.molecule_id.to_numpy(dtype=np.int64)
            x2_task = x2[molecule_ids]
            x3_task = x3[molecule_ids]
            x_fused_task = np.concatenate([x2_task, x3_task], axis=1)
            inner_kernels = [precompute_tanimoto_split(
                k2, molecule_ids, tr[fit_rel], tr[val_rel])
                for fit_rel, val_rel in inner_splits]
            outer_kernels = precompute_tanimoto_split(k2, molecule_ids, tr, te)
            families = make_candidates(args.n_estimators, 20260925 + int(outer_fold),
                                       x2_task.shape[1], x3_task.shape[1])
            fit_means = part.iloc[tr].groupby("assay_chembl_id").y.mean()
            global_mean = float(part.iloc[tr].y.mean())
            context_means = part.iloc[te].assay_chembl_id.map(fit_means)
            context_seen = context_means.notna().to_numpy()
            context_pred = context_means.fillna(global_mean).to_numpy(dtype=float)

            def score_inner_family(candidates, family_name):
                scored = []
                for name, kind, params, view in candidates:
                    x_view = {"2d": x2_task, "3d": x3_task}.get(view)
                    oof = np.full(len(tr), np.nan, dtype=np.float64)
                    known = np.zeros(len(tr), dtype=bool)
                    for inner_i, (fit_rel, val_rel) in enumerate(inner_splits):
                        fit_idx, val_idx = tr[fit_rel], tr[val_rel]
                        pred, seen, _ = kernel_base.fit_assay_kernel_predict(
                            kind, params, x_view, part, fit_idx, val_idx,
                            inner_kernels[inner_i], 20260925 + int(outer_fold) * 100 + inner_i)
                        oof[val_rel] = pred
                        known[val_rel] = seen
                    if not np.isfinite(oof).all() or not known.any():
                        raise RuntimeError(f"Incomplete inner OOF: {task}/{outer_fold}/{name}")
                    score = float(np.sqrt(mean_squared_error(
                        part.iloc[tr].y.to_numpy()[known], oof[known])))
                    candidate_rows.append({"task": task, "outer_fold": int(outer_fold),
                                           "model": family_name, "candidate": name,
                                           "kind": kind, "feature_view": view,
                                           "inner_oof_rmse": score})
                    scored.append({"score": score, "name": name, "kind": kind,
                                   "params": params, "view": view,
                                   "oof": oof, "known": known})
                    print(f"{task} outer={outer_fold + 1}/5 {family_name} {name} inner_rmse={score:.4f}", flush=True)
                return scored

            scored_2d = score_inner_family(families[MODEL_2D], MODEL_2D)
            best_2d = min(scored_2d, key=lambda item: item["score"])
            scored_3d = score_inner_family(families[MODEL_3D_AWARE], MODEL_3D_AWARE)
            blend_candidates = []
            for candidate in scored_3d:
                if not np.array_equal(candidate["known"], best_2d["known"]):
                    raise RuntimeError("2D and 3D inner assay support differs")
                for weight_3d in (0.1, 0.25, 0.5):
                    blend_oof = ((1.0 - weight_3d) * best_2d["oof"]
                                 + weight_3d * candidate["oof"])
                    known = best_2d["known"]
                    score = float(np.sqrt(mean_squared_error(
                        part.iloc[tr].y.to_numpy()[known], blend_oof[known])))
                    name = f"late_blend_{best_2d['name']}__{candidate['name']}_w{weight_3d:g}"
                    candidate_rows.append({"task": task, "outer_fold": int(outer_fold),
                                           "model": MODEL_3D_AWARE, "candidate": name,
                                           "kind": "late_blend", "feature_view": "late_fusion",
                                           "inner_oof_rmse": score})
                    blend_candidates.append({"score": score, "name": name,
                                             "kind": "late_blend",
                                             "params": (best_2d["name"], candidate["name"], weight_3d),
                                             "view": "late_fusion", "oof": blend_oof,
                                             "known": known})
                    print(f"{task} outer={outer_fold + 1}/5 {MODEL_3D_AWARE} {name} inner_rmse={score:.4f}", flush=True)

            best_3d = min(scored_3d + blend_candidates, key=lambda item: item["score"])
            base_2d_info = next(c for c in families[MODEL_2D] if c[0] == best_2d["name"])
            pred_2d, seen_2d, _ = kernel_base.fit_assay_kernel_predict(
                base_2d_info[1], base_2d_info[2],
                {"2d": x2_task}.get(base_2d_info[3]), part, tr, te, outer_kernels,
                20260925 + int(outer_fold))
            if best_3d["kind"] == "late_blend":
                selected_2d_name, selected_3d_name, weight_3d = best_3d["params"]
                if selected_2d_name != best_2d["name"]:
                    raise RuntimeError("Late-fusion baseline differs from the selected 2D model")
                three_d_info = next(c for c in families[MODEL_3D_AWARE] if c[0] == selected_3d_name)
                pred_3d, seen_3d, _ = kernel_base.fit_assay_kernel_predict(
                    three_d_info[1], three_d_info[2], x3_task, part, tr, te, outer_kernels,
                    20260925 + int(outer_fold))
                if not np.array_equal(seen_2d, seen_3d):
                    raise RuntimeError("Outer 2D and 3D assay support differs")
                pred_aware = (1.0 - weight_3d) * pred_2d + weight_3d * pred_3d
                aware_seen = seen_2d
                aware_kind = "late_blend"
                aware_view = "late_fusion"
            else:
                pred_aware, aware_seen, _ = kernel_base.fit_assay_kernel_predict(
                    best_3d["kind"], best_3d["params"], x3_task, part, tr, te, outer_kernels,
                    20260925 + int(outer_fold))
                aware_kind = best_3d["kind"]
                aware_view = best_3d["view"]
            if not np.array_equal(seen_2d, context_seen) or not np.array_equal(aware_seen, context_seen):
                raise RuntimeError("Assay support differs from assay-mean-only comparator")

            selection_rows.extend([
                {"task": task, "outer_fold": int(outer_fold), "model": MODEL_2D,
                 "selected_candidate": best_2d["name"], "selected_kind": best_2d["kind"],
                 "selected_feature_view": best_2d["view"], "inner_oof_rmse": best_2d["score"]},
                {"task": task, "outer_fold": int(outer_fold), "model": MODEL_3D_AWARE,
                 "selected_candidate": best_3d["name"], "selected_kind": aware_kind,
                 "selected_feature_view": aware_view, "inner_oof_rmse": best_3d["score"]},
            ])
            for model_name, pred, seen in ((MODEL_2D, pred_2d, seen_2d),
                                           (MODEL_3D_AWARE, pred_aware, aware_seen)):
                for j, idx in enumerate(te):
                    row = part.iloc[idx]
                    pred_rows.append({"record_id": int(row.record_id), "task": task,
                                      "outer_fold": int(outer_fold), "assay_chembl_id": str(row.assay_chembl_id),
                                      "smiles": str(row.smiles), "scaffold": str(row.scaffold),
                                      "y_true": float(row.y), "assay_train_n": int((part.iloc[tr].assay_chembl_id == row.assay_chembl_id).sum()),
                                      "assay_seen_in_training": bool(seen[j]),
                                      "model": model_name, "y_pred": float(pred[j])})
            for j, idx in enumerate(te):
                row = part.iloc[idx]
                pred_rows.append({"record_id": int(row.record_id), "task": task,
                                  "outer_fold": int(outer_fold), "assay_chembl_id": str(row.assay_chembl_id),
                                  "smiles": str(row.smiles), "scaffold": str(row.scaffold),
                                  "y_true": float(row.y), "assay_train_n": int((part.iloc[tr].assay_chembl_id == row.assay_chembl_id).sum()),
                                  "assay_seen_in_training": bool(context_seen[j]),
                                  "model": MODEL_CONTEXT, "y_pred": float(context_pred[j])})
            pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
            pd.DataFrame(candidate_rows).to_csv(cand_path, index=False)
            pd.DataFrame(selection_rows).to_csv(select_path, index=False)
            completed.add((task, int(outer_fold)))

    preds = pd.DataFrame(pred_rows)
    summaries = []
    paired_rows = []
    rng = np.random.default_rng(20260925)
    for task, task_preds in preds.groupby("task"):
        by_model = {model: group.set_index("record_id") for model, group in task_preds.groupby("model")}
        for model in (MODEL_2D, MODEL_3D_AWARE, MODEL_CONTEXT):
            group = task_preds.loc[task_preds.model == model]
            known = group.loc[group.assay_seen_in_training]
            unknown = group.loc[~group.assay_seen_in_training]
            summaries.append({"task": task, "model": model, **metric_row(group.y_true.to_numpy(), group.y_pred.to_numpy()),
                              "known_assay_coverage": float(group.assay_seen_in_training.mean()),
                              "known_assay_rmse": metric_row(known.y_true.to_numpy(), known.y_pred.to_numpy())["rmse"],
                              "unseen_assay_rmse": metric_row(unknown.y_true.to_numpy(), unknown.y_pred.to_numpy())["rmse"] if len(unknown) else float("nan"),
                              "n_assays_tested": int(group.assay_chembl_id.nunique()),
                              "n_unique_molecules": int(group.smiles.nunique())})
        baseline = by_model[MODEL_2D]
        for candidate in (MODEL_3D_AWARE, MODEL_CONTEXT):
            comp = by_model[candidate]
            paired = baseline.loc[baseline.assay_seen_in_training, ["scaffold", "y_true", "y_pred"]].join(
                comp.loc[comp.assay_seen_in_training, ["y_true", "y_pred"]].rename(
                    columns={"y_true": "candidate_y", "y_pred": "candidate_pred"}),
                how="inner", validate="one_to_one")
            if len(paired) != int(baseline.assay_seen_in_training.sum()) or not np.allclose(
                    paired.y_true.to_numpy(), paired.candidate_y.to_numpy(), atol=1e-7, rtol=0):
                raise RuntimeError(f"Could not pair exact records for {task}/{candidate}")
            y = paired.y_true.to_numpy()
            p2 = paired.y_pred.to_numpy()
            p3 = paired.candidate_pred.to_numpy()
            scaffolds = paired.scaffold.unique()
            by_scaffold = {s: np.flatnonzero(paired.scaffold.to_numpy() == s) for s in scaffolds}
            deltas = []
            for _ in range(2000):
                chosen = rng.choice(scaffolds, size=len(scaffolds), replace=True)
                idx = np.concatenate([by_scaffold[s] for s in chosen])
                deltas.append(float(np.sqrt(np.mean((p3[idx]-y[idx])**2))
                                       - np.sqrt(np.mean((p2[idx]-y[idx])**2))))
            lo, hi = np.quantile(deltas, [0.025, 0.975])
            paired_rows.append({"task": task, "candidate": candidate, "baseline": MODEL_2D,
                                "n_paired_known_assay_observations": int(len(paired)),
                                "delta_candidate_minus_2d_rmse": float(np.sqrt(np.mean((p3-y)**2))
                                                                           - np.sqrt(np.mean((p2-y)**2))),
                                "scaffold_bootstrap_ci95_low": float(lo),
                                "scaffold_bootstrap_ci95_high": float(hi),
                                "bootstrap_probability_candidate_better": float(np.mean(np.asarray(deltas)<0)),
                                "scaffold_groups": int(len(scaffolds))})

    pd.DataFrame(summaries).to_csv(out_dir / "pooled_assay_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out_dir / "paired_scaffold_bootstrap.csv", index=False)
    preds.to_csv(pred_path, index=False)
    pd.DataFrame(candidate_rows).to_csv(cand_path, index=False)
    pd.DataFrame(selection_rows).to_csv(select_path, index=False)
    audit = {
        "status": "smoke_only" if args.max_outer_folds else "complete",
        "source_file": str(SOURCE.relative_to(ROOT)), "source_sha256": assay_base.sha256(SOURCE),
        "feature_2d_sha256": assay_base.sha256(FEAT2),
        "feature_3d_file": str(feature_path.relative_to(ROOT) if feature_path.is_relative_to(ROOT) else feature_path),
        "feature_3d_sha256": assay_base.sha256(feature_path),
        "fold_manifest": str(manifest_path.relative_to(ROOT) if manifest_path.is_relative_to(ROOT) else manifest_path),
        "fold_manifest_sha256": assay_base.sha256(manifest_path),
        "validation": f"nested {args.split_level} GroupKFold; train-only assay means; fold-local scaler/PCA and late-fusion-weight selection inside inner known-assay RMSE",
        "model_protocol": "Stage10 shrinkage late fusion; PCA fitted only on current fit fold; 3D PCA ranks 16/32/64 with RBF-SVR C 0.1/1; 2D/3D blend weights 0.1/0.25/0.5 selected on exact inner OOF records",
        "split_level": args.split_level,
        "outer_scaffold_overlap": "required zero" if args.split_level == "scaffold" else "allowed; random molecule holdout tests within-scaffold interpolation",
        "outer_molecule_overlap": 0,
        "min_assay_unique_molecules": int(args.min_assay_molecules),
        "min_rotatable_bonds": int(args.min_rotatable_bonds),
        "domain_scope": "full matched cohort" if args.min_rotatable_bonds == 0 else f"molecules with at least {args.min_rotatable_bonds} RDKit rotatable bonds",
        "feature_views": {"2d": list(x2.shape), "3d": list(x3.shape), "hybrid_dimension": int(x2.shape[1]+x3.shape[1])},
        "warnings": ["Internal ChEMBL CV only; no prospective validation.",
                     "No raw high-dimensional 3D features or 3D kernels are candidate models in Stage10.",
                     "This task-specific strategy is confirmatory only on the named independent fold manifest."]
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(paired_rows).to_string(index=False))


if __name__ == "__main__":
    main()
