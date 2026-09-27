"""Select a 3D-bearing model family using inner OOF only, then audit outer OOF.

For each task and outer scaffold fold, the 3D-aware family is the union of
assay-centered Uni-Mol2/RDKit-3D-only and 2D+Uni-Mol2/RDKit-3D candidates.
The family/candidate choice is made exclusively from inner OOF RMSE. Outer
predictions are opened only after the selection table is frozen in memory.
The paired primary estimate is restricted to assays represented in training,
matching the benchmark's known-assay estimand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_2D = "assay_centered_2d"
MODEL_3D = "assay_centered_3d"
MODEL_FUSED = "assay_centered_2d_plus_3d"
KEYS = ["task", "outer_fold", "assay_chembl_id", "smiles"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Raw benchmark output directory")
    ap.add_argument("--output-dir", required=True, help="Selected-method audit output")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()
    run_dir, out_dir = Path(args.run_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_path = run_dir / "inner_candidate_scores.csv"
    selection_path = run_dir / "inner_model_selections.csv"
    pred_path = run_dir / "outer_oof_predictions.csv"
    scores = pd.read_csv(score_path)
    # Freeze family and candidate choices from inner scores before reading the
    # outer predictions. Ties favor the standalone 3D model (simpler family).
    eligible = scores.loc[scores.model.isin([MODEL_3D, MODEL_FUSED])].copy()
    eligible["tie_priority"] = eligible.model.map({MODEL_3D: 0, MODEL_FUSED: 1})
    family_best = (eligible.sort_values(
        ["task", "outer_fold", "inner_oof_rmse", "tie_priority", "candidate"],
        ascending=True, kind="mergesort")
        .groupby(["task", "outer_fold"], as_index=False).first())
    if family_best.empty:
        raise RuntimeError("No eligible 3D-bearing inner candidates")
    expected = set(scores.loc[scores.model == MODEL_2D, ["task", "outer_fold"]]
                   .itertuples(index=False, name=None))
    observed = set(family_best[["task", "outer_fold"]].itertuples(index=False, name=None))
    if expected != observed:
        raise RuntimeError(f"3D family selection coverage mismatch: {expected ^ observed}")
    family_best = family_best.rename(columns={
        "model": "selected_3d_bearing_family",
        "candidate": "selected_candidate",
        "inner_oof_rmse": "selection_inner_oof_rmse",
    })
    family_best.to_csv(out_dir / "inner_family_selections.csv", index=False)

    # The outer file is consulted only after validation-only family selection.
    preds = pd.read_csv(pred_path)
    baseline = preds.loc[preds.model == MODEL_2D].copy()
    baseline_select = pd.read_csv(selection_path)
    baseline_select = baseline_select.loc[baseline_select.model == MODEL_2D,
        ["task", "outer_fold", "selected_candidate", "inner_oof_rmse"]].rename(
            columns={"selected_candidate": "selected_2d_candidate",
                     "inner_oof_rmse": "selected_2d_inner_oof_rmse"})

    chunks = []
    for row in family_best.itertuples(index=False):
        chosen = preds.loc[(preds.task == row.task)
                           & (preds.outer_fold == row.outer_fold)
                           & (preds.model == row.selected_3d_bearing_family)].copy()
        if chosen.empty:
            raise RuntimeError(f"Missing selected 3D-bearing prediction: {row.task}/{row.outer_fold}")
        chosen["selected_3d_candidate"] = row.selected_candidate
        chosen["selected_3d_family_inner_oof_rmse"] = row.selection_inner_oof_rmse
        chunks.append(chosen)
    candidate = pd.concat(chunks, ignore_index=True)
    left = baseline.set_index(KEYS).sort_index()
    right = candidate.set_index(KEYS).sort_index()
    if not left.index.equals(right.index):
        raise RuntimeError("Selected 3D-aware and 2D OOF rows do not align exactly")
    for col in ["y_true", "assay_seen_in_training", "scaffold"]:
        if pd.api.types.is_numeric_dtype(left[col].dtype):
            same = np.allclose(left[col].to_numpy(dtype=float),
                               right[col].to_numpy(dtype=float), atol=1e-7, rtol=0,
                               equal_nan=True)
        else:
            same = np.array_equal(left[col].fillna("<NA>").astype(str).to_numpy(),
                                  right[col].fillna("<NA>").astype(str).to_numpy())
        if not same:
            raise RuntimeError(f"Paired OOF mismatch in {col}")
    paired = left[["y_true", "y_pred", "assay_seen_in_training", "scaffold"]].rename(
        columns={"y_pred": "y_pred_2d"}).join(
        right[["y_pred", "selected_3d_candidate", "selected_3d_family_inner_oof_rmse"]]
        .rename(columns={"y_pred": "y_pred_3d_aware"}))
    paired = paired.reset_index().merge(baseline_select, on=["task", "outer_fold"],
                                         how="left", validate="many_to_one")
    paired = paired.merge(family_best[["task", "outer_fold", "selected_3d_bearing_family",
                                       "selection_inner_oof_rmse"]],
                          on=["task", "outer_fold"], how="left", validate="many_to_one")
    paired.to_csv(out_dir / "paired_outer_oof_predictions.csv", index=False)

    summaries, folds, intervals = [], [], []
    for task_idx, (task, task_part) in enumerate(paired.groupby("task", sort=True)):
        known = task_part.loc[task_part.assay_seen_in_training.astype(bool)].copy()
        if known.empty:
            raise RuntimeError(f"No known-assay paired test rows for {task}")
        y = known.y_true.to_numpy(dtype=float)
        p2 = known.y_pred_2d.to_numpy(dtype=float)
        p3 = known.y_pred_3d_aware.to_numpy(dtype=float)
        delta = rmse(y, p3) - rmse(y, p2)
        summaries.append({
            "task": task, "n_known_assay_records": len(known),
            "n_unique_molecules": int(known.smiles.nunique()),
            "n_scaffolds": int(known.scaffold.fillna("").nunique()),
            "known_assay_coverage_all_outer_rows": float(task_part.assay_seen_in_training.mean()),
            "rmse_2d": rmse(y, p2), "rmse_selected_3d_aware": rmse(y, p3),
            "delta_rmse_3d_aware_minus_2d": delta,
            "mae_2d": float(mean_absolute_error(y, p2)),
            "mae_selected_3d_aware": float(mean_absolute_error(y, p3)),
            "r2_2d": float(r2_score(y, p2)), "r2_selected_3d_aware": float(r2_score(y, p3)),
        })
        for fold, fold_part in known.groupby("outer_fold", sort=True):
            yf = fold_part.y_true.to_numpy(dtype=float)
            f2 = fold_part.y_pred_2d.to_numpy(dtype=float)
            f3 = fold_part.y_pred_3d_aware.to_numpy(dtype=float)
            folds.append({"task": task, "outer_fold": int(fold), "n": len(fold_part),
                          "selected_3d_bearing_family": str(fold_part.selected_3d_bearing_family.iloc[0]),
                          "selected_3d_candidate": str(fold_part.selected_3d_candidate.iloc[0]),
                          "selected_3d_inner_oof_rmse": float(fold_part.selection_inner_oof_rmse.iloc[0]),
                          "selected_2d_candidate": str(fold_part.selected_2d_candidate.iloc[0]),
                          "selected_2d_inner_oof_rmse": float(fold_part.selected_2d_inner_oof_rmse.iloc[0]),
                          "rmse_2d": rmse(yf, f2), "rmse_3d_aware": rmse(yf, f3),
                          "delta_rmse": rmse(yf, f3) - rmse(yf, f2)})

        # Acyclic molecules have an empty Murcko scaffold; keep them together
        # as the same explicit group used by the split generator.
        scaffold_ids = known.scaffold.fillna("").astype(str).to_numpy()
        unique_scaffolds = np.unique(scaffold_ids)
        by_scaffold = {s: np.flatnonzero(scaffold_ids == s) for s in unique_scaffolds}
        rng = np.random.default_rng(args.seed + task_idx)
        boot_delta = np.empty(args.bootstrap, dtype=float)
        for b in range(args.bootstrap):
            draw = rng.choice(unique_scaffolds, size=len(unique_scaffolds), replace=True)
            ix = np.concatenate([by_scaffold[s] for s in draw])
            boot_delta[b] = rmse(y[ix], p3[ix]) - rmse(y[ix], p2[ix])
        lo, hi = np.quantile(boot_delta, [0.025, 0.975])
        intervals.append({"task": task, "candidate": "validation_selected_3d_bearing_family",
                          "baseline": MODEL_2D, "n_records": len(known),
                          "scaffold_groups": len(unique_scaffolds),
                          "delta_rmse_3d_aware_minus_2d": delta,
                          "scaffold_bootstrap_ci95_low": float(lo),
                          "scaffold_bootstrap_ci95_high": float(hi),
                          "bootstrap_probability_3d_aware_better": float(np.mean(boot_delta < 0)),
                          "bootstrap_replicates": args.bootstrap})

    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(folds).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(intervals).to_csv(out_dir / "paired_scaffold_bootstrap.csv", index=False)
    source_audit_path = run_dir / "audit.json"
    source_audit = json.loads(source_audit_path.read_text()) if source_audit_path.exists() else {}
    audit = {
        "status": "complete", "source_run_dir": str(run_dir),
        "source_inner_candidate_scores_sha256": sha256(score_path),
        "source_outer_predictions_sha256": sha256(pred_path),
        "source_audit": source_audit,
        "selection_rule": "per task and outer fold, select minimum inner OOF RMSE among assay-centered 3D-only and 2D+3D candidates; tie favors 3D-only",
        "outer_predictions_used_for_selection": False,
        "primary_estimand": "known-assay scaffold-held-out activity prediction",
        "paired_rows_exactly_aligned": True,
        "task_specific_nested_selection": True,
        "caveats": ["Internal ChEMBL cross-validation; not external or prospective validation.",
                     "3D-aware family may select either standalone 3D or 2D+3D, and is reported as such.",
                     "Primary comparison is restricted to assays present in the corresponding outer training split."],
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(pd.DataFrame(intervals).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
