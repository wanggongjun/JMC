#!/usr/bin/env python3
"""Established per-cell-line assay-aware 2D reference on a shared scaffold manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import assay_context_3d_benchmark as base
import assay_context_kernel_benchmark as kernel

ROOT = base.ROOT
MODEL = "established_independent_2d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=100)
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

    raw = pd.read_csv(base.SOURCE, encoding="utf-8-sig")
    raw = raw.loc[(raw.standard_type.astype(str).str.upper() == "IC50")
                  & (raw.standard_relation.astype(str) == "=")
                  & raw.pchembl_value_final.notna()].copy()
    raw["task"] = raw.cell_line.map(base.TASKS)
    raw = raw.dropna(subset=["task", "assay_chembl_id", "canonical_smiles"]).copy()
    raw["smiles"] = raw.canonical_smiles.astype(str)
    raw["y"] = pd.to_numeric(raw.pchembl_value_final, errors="coerce")
    raw = raw.dropna(subset=["y"])
    feat = np.load(base.FEAT2, allow_pickle=False)
    feature_map = {str(s): i for i, s in enumerate(feat["smiles"])}
    raw = raw.loc[raw.smiles.isin(feature_map)].copy()
    counts = raw.groupby(["task", "assay_chembl_id"]).smiles.nunique().rename("n").reset_index()
    kept = counts.loc[counts.n >= 10, ["task", "assay_chembl_id"]]
    raw = raw.merge(kept, on=["task", "assay_chembl_id"], how="inner", validate="many_to_one")
    folds = pd.read_csv(manifest_path)
    fold_by_smiles = folds.drop_duplicates("smiles").set_index("smiles").outer_fold.to_dict()
    scaffold_by_smiles = folds.drop_duplicates("smiles").set_index("smiles").scaffold.to_dict()
    raw = raw.loc[raw.smiles.isin(fold_by_smiles)].copy().reset_index(drop=True)
    raw["outer_fold"] = raw.smiles.map(fold_by_smiles).astype(int)
    raw["scaffold"] = raw.smiles.map(scaffold_by_smiles).astype(str)
    raw["record_id"] = np.arange(len(raw), dtype=np.int64)

    molecules = raw.smiles.drop_duplicates().astype(str).tolist()
    molecule_map = {s: i for i, s in enumerate(molecules)}
    row_molecule_ids = raw.smiles.map(molecule_map).to_numpy(dtype=np.int64)
    x2 = feat["features"][np.asarray([feature_map[s] for s in molecules])].astype(np.float32)
    if not np.isfinite(x2).all():
        raise RuntimeError("2D feature matrix has non-finite values")
    with threadpool_limits(limits=1, user_api="blas"):
        k2 = kernel.tanimoto_kernel(x2)

    prediction_rows = []
    selection_rows = []
    candidate_rows = []
    outer_folds = sorted(raw.outer_fold.unique().tolist())
    if args.max_outer_folds:
        outer_folds = outer_folds[:args.max_outer_folds]
    tasks = sorted(raw.task.unique().tolist())
    for task in tasks:
        frame = raw.loc[raw.task == task].reset_index(drop=True)
        row_folds = frame.outer_fold.to_numpy(dtype=int)
        row_mol_ids = frame.smiles.map(molecule_map).to_numpy(dtype=np.int64)
        x2_rows = x2[row_mol_ids]
        groups = frame.scaffold.astype(str).to_numpy()
        for outer_fold in outer_folds:
            tr = np.flatnonzero(row_folds != outer_fold)
            te = np.flatnonzero(row_folds == outer_fold)
            inner = GroupKFold(n_splits=args.inner_folds)
            inner_splits = list(inner.split(tr, frame.y.to_numpy()[tr], groups[tr]))
            inner_kernels = []
            for fit_rel, val_rel in inner_splits:
                fit_mols, val_mols = row_mol_ids[tr[fit_rel]], row_mol_ids[tr[val_rel]]
                inner_kernels.append({"tanimoto": (k2[np.ix_(fit_mols, fit_mols)],
                                                     k2[np.ix_(val_mols, fit_mols)])})
            candidates = [(name, "vector2", proto, x2_rows) for name, proto in
                          base.candidates("2d", 20261000 + int(outer_fold), args.n_estimators)]
            candidates.extend((f"tanimoto_krr_a{a:g}", "tanimoto", float(a), None)
                              for a in (0.01, 0.1, 1.0, 10.0))
            scores = []
            for name, kind, params, features in candidates:
                oof = np.full(len(tr), np.nan, dtype=np.float64)
                known = np.zeros(len(tr), dtype=bool)
                for split_i, (fit_rel, val_rel) in enumerate(inner_splits):
                    fit_idx, val_idx = tr[fit_rel], tr[val_rel]
                    fit_pred, seen, _ = kernel.fit_assay_kernel_predict(
                        kind, params, features, frame, fit_idx, val_idx,
                        inner_kernels[split_i], 20261000 + int(outer_fold)*100 + split_i)
                    oof[val_rel] = fit_pred
                    known[val_rel] = seen
                if not np.isfinite(oof).all() or not known.any():
                    raise RuntimeError(f"Incomplete inner OOF for {task}/{outer_fold}/{name}")
                score = float(np.sqrt(mean_squared_error(frame.y.to_numpy()[tr][known], oof[known])))
                candidate_rows.append({"task": task, "outer_fold": int(outer_fold),
                                       "candidate": name, "inner_known_assay_rmse": score,
                                       "inner_known_assay_n": int(known.sum())})
                scores.append((score, name, kind, params, features))
                print(f"{task} outer={int(outer_fold)+1}/5 {name} inner_rmse={score:.4f}", flush=True)
            best = min(scores, key=lambda item: item[0])
            outer_kernels = {"tanimoto": (k2[np.ix_(row_mol_ids[tr], row_mol_ids[tr])],
                                           k2[np.ix_(row_mol_ids[te], row_mol_ids[tr])])}
            pred, seen, _ = kernel.fit_assay_kernel_predict(
                best[2], best[3], best[4], frame, tr, te, outer_kernels,
                20262000 + int(outer_fold))
            context_mean = frame.iloc[tr].groupby("assay_chembl_id").y.mean()
            global_mean = float(frame.iloc[tr].y.mean())
            ctx = frame.iloc[te].assay_chembl_id.map(context_mean).to_numpy(dtype=np.float64)
            ctx_seen = np.isfinite(ctx)
            ctx[~ctx_seen] = global_mean
            selection_rows.append({"task": task, "outer_fold": int(outer_fold),
                                   "selected_candidate": best[1],
                                   "inner_known_assay_rmse": float(best[0])})
            for model_name, values, mask in ((MODEL, pred, seen),
                                              ("assay_mean_only", ctx, ctx_seen)):
                for j, idx in enumerate(te):
                    row = frame.iloc[idx]
                    prediction_rows.append({"record_id": int(row.record_id), "task": task,
                                            "outer_fold": int(outer_fold),
                                            "assay_chembl_id": str(row.assay_chembl_id),
                                            "smiles": str(row.smiles), "scaffold": str(row.scaffold),
                                            "y_true": float(row.y), "known_assay": bool(mask[j]),
                                            "model": model_name, "y_pred": float(values[j])})
            pd.DataFrame(prediction_rows).to_csv(out / "outer_oof_predictions.csv", index=False)
            pd.DataFrame(selection_rows).to_csv(out / "inner_model_selections.csv", index=False)
            pd.DataFrame(candidate_rows).to_csv(out / "inner_candidate_scores.csv", index=False)
            print(f"completed {task} outer fold {int(outer_fold)+1}/5", flush=True)

    pred_frame = pd.DataFrame(prediction_rows)
    summary = []
    for (task, model_name), part in pred_frame.groupby(["task", "model"]):
        known = part.loc[part.known_assay]
        summary.append({"task": task, "model": model_name, "n": int(len(part)),
                        "known_assay_coverage": float(part.known_assay.mean()),
                        "known_assay_n": int(len(known)),
                        "known_assay_rmse": float(np.sqrt(mean_squared_error(known.y_true, known.y_pred))),
                        "known_assay_mae": float(mean_absolute_error(known.y_true, known.y_pred)),
                        "known_assay_r2": float(r2_score(known.y_true, known.y_pred))})
    pred_frame.to_csv(out / "outer_oof_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "pooled_assay_metrics.csv", index=False)
    audit = {"status": "smoke_only" if args.max_outer_folds else "complete",
             "source_sha256": sha256(base.SOURCE), "features_2d_sha256": sha256(base.FEAT2),
             "manifest_sha256": sha256(manifest_path),
             "validation": "nested 4-fold scaffold GroupKFold; fold manifest scaffold assignment shared across tasks; separate per-cell-line 2D selectors",
             "candidate_family": "existing assay-aware 2D SVR, HistGradientBoosting, ExtraTrees and Morgan-Tanimoto KRR grid",
             "outer_molecule_overlap": 0, "cross_task_scaffold_leakage": 0,
             "warnings": ["Internal ChEMBL scaffold CV only; no prospective biological validation."]}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
