#!/usr/bin/env python3
"""Pair Stage 12 Uni-Mol2 OOF rows against both 2D references and audit the gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/optimization_v1"
TASKS = ("hepg2_pIC50", "hct116_pIC50")
SEEDS = ("20261009", "20261010")
MODEL_2D = "assay_centered_2d_reference"
MODEL_3D = "assay_centered_3d_aware_strategy"
MODEL_CONTEXT = "assay_mean_only"
MODEL_INDEP = "established_independent_2d"
KEYS = ["record_id"]


def scaffold_bootstrap(paired: pd.DataFrame, seed: int, n_boot: int):
    y = paired.y_true.to_numpy(dtype=np.float64)
    baseline = paired.y_baseline.to_numpy(dtype=np.float64)
    candidate = paired.y_candidate.to_numpy(dtype=np.float64)
    groups = paired.scaffold.astype(str).to_numpy()
    unique = np.unique(groups)
    row_groups = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    delta = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([row_groups[group] for group in picked])
        delta[b] = (np.sqrt(np.mean((candidate[idx]-y[idx])**2))
                    - np.sqrt(np.mean((baseline[idx]-y[idx])**2)))
    return {
        "n_paired_known_assay_records": int(len(paired)),
        "delta_3d_minus_2d_rmse": float(np.sqrt(np.mean((candidate-y)**2))
                                         - np.sqrt(np.mean((baseline-y)**2))),
        "scaffold_bootstrap_ci95_low": float(np.quantile(delta, .025)),
        "scaffold_bootstrap_ci95_high": float(np.quantile(delta, .975)),
        "bootstrap_probability_3d_better": float(np.mean(delta < 0)),
        "n_scaffolds": int(len(unique)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--output-dir", default="experiments/optimization_v1/stage12_result_audit")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    metrics, comparisons, selections, validations, oof_rows = [], [], [], [], []
    gate_pass = True
    for seed in SEEDS:
        manifest = pd.read_csv(BASE / f"shared_scaffold_folds_seed_{seed}.csv")
        if manifest.groupby("scaffold").outer_fold.nunique().max() != 1:
            raise RuntimeError(f"Shared-scaffold assignment is inconsistent: seed {seed}")
        stage_manifest = manifest.drop_duplicates("smiles").set_index("smiles").outer_fold.to_dict()
        joint_dir = BASE / f"stage12_fusion_seed_{seed}"
        independent_dir = BASE / f"stage12_2d_seed_{seed}"
        for directory in (joint_dir, independent_dir):
            audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
            if audit.get("status") != "complete":
                raise RuntimeError(f"Incomplete/smoke-only result: {directory}")
            if audit.get("outer_molecule_overlap") != 0:
                raise RuntimeError(f"Molecule leakage audit failed: {directory}")
        stage = pd.read_csv(joint_dir / "outer_oof_predictions.csv")
        stage["known_assay"] = stage.assay_seen_in_training.astype(bool)
        independent = pd.read_csv(independent_dir / "outer_oof_predictions.csv")
        independent["known_assay"] = independent.known_assay.astype(bool)

        selected_models = set(stage.model.unique())
        if selected_models != {MODEL_2D, MODEL_3D, MODEL_CONTEXT}:
            raise RuntimeError(f"Unexpected Stage 12 models for {seed}: {selected_models}")
        if set(independent.model.unique()) != {MODEL_INDEP, MODEL_CONTEXT}:
            raise RuntimeError(f"Unexpected independent 2D models for {seed}")
        stage_2d = stage.loc[stage.model == MODEL_2D].copy()
        stage_3d = stage.loc[stage.model == MODEL_3D].copy()
        independent_2d = independent.loc[independent.model == MODEL_INDEP].copy()
        for name, frame in (("stage12-2d", stage_2d), ("stage12-3d", stage_3d),
                            ("independent-2d", independent_2d)):
            if not np.isfinite(frame.y_pred).all():
                raise RuntimeError(f"Non-finite predictions in {name}/{seed}")
            if frame.record_id.duplicated().any():
                raise RuntimeError(f"Duplicate OOF keys in {name}/{seed}")
            if any(stage_manifest.get(str(s), -1) != int(f)
                   for s, f in zip(frame.smiles, frame.outer_fold)):
                raise RuntimeError(f"OOF fold key differs from shared manifest in {name}/{seed}")

        selection = pd.read_csv(joint_dir / "inner_model_selections.csv")
        selections.extend(selection.assign(seed=seed).to_dict("records"))
        for task in TASKS:
            candidate = stage_3d.loc[stage_3d.task == task].copy()
            for source, baseline_name, baseline_model, baseline_known in (
                    (stage_2d, "matched_nested_2d", MODEL_2D, "assay_seen_in_training"),
                    (independent_2d, "established_independent_2d", MODEL_INDEP, "known_assay")):
                baseline = source.loc[source.task == task].copy()
                candidate_known = candidate.loc[candidate.known_assay].copy()
                baseline_known_rows = baseline.loc[baseline[baseline_known].astype(bool)].copy()
                # Match by the preserved source-row ID because this ChEMBL file
                # contains two distinct labels for one task/assay/SMILES key.
                check_cols = ["task", "outer_fold", "assay_chembl_id", "smiles"]
                left = baseline_known_rows[KEYS + check_cols + ["scaffold", "y_true", "y_pred"]].rename(
                    columns={"y_pred": "y_baseline"})
                right = candidate_known[KEYS + check_cols + ["scaffold", "y_true", "y_pred"]].rename(
                    columns={"y_pred": "y_candidate", "scaffold": "scaffold_candidate",
                             "y_true": "y_candidate_label", **{c: f"{c}_candidate" for c in check_cols}})
                paired = left.merge(right, on=KEYS, how="inner", validate="one_to_one")
                if len(paired) != len(left) or len(right) != len(left):
                    raise RuntimeError(f"Known-assay OOF support mismatch: {seed}/{task}/{baseline_name}")
                if not np.allclose(paired.y_true, paired.y_candidate_label, atol=1e-7, rtol=0):
                    raise RuntimeError(f"Target values mismatch: {seed}/{task}/{baseline_name}")
                # Empty Murcko scaffolds are serialized as blank CSV cells and
                # re-read as NaN; compare their canonical string form.
                if not np.array_equal(paired.scaffold.astype(str), paired.scaffold_candidate.astype(str)):
                    raise RuntimeError(f"Scaffold keys mismatch: {seed}/{task}/{baseline_name}")
                for col in check_cols:
                    if not np.array_equal(paired[col].astype(str), paired[f"{col}_candidate"].astype(str)):
                        raise RuntimeError(f"Source-row metadata mismatch in {col}: {seed}/{task}/{baseline_name}")
                result = scaffold_bootstrap(
                    paired, seed=20264000 + int(seed) + (0 if task == TASKS[0] else 100)
                    + (0 if baseline_model == MODEL_2D else 10), n_boot=args.n_bootstrap)
                row = {"seed": seed, "task": task, "baseline": baseline_name,
                       "candidate": MODEL_3D, **result}
                comparisons.append(row)
                pass_pair = result["delta_3d_minus_2d_rmse"] < 0
                gate_pass &= pass_pair
                validations.append({"seed": seed, "task": task, "baseline": baseline_name,
                                    "known_assay_n": result["n_paired_known_assay_records"],
                                    "point_estimate_pass": pass_pair})
                oof_rows.extend(paired.assign(seed=seed, baseline=baseline_name).to_dict("records"))

            for model_name, group in (("matched_nested_2d", stage_2d.loc[stage_2d.task == task]),
                                      ("stage12_2d_plus_unimol2", candidate),
                                      ("established_independent_2d", independent_2d.loc[independent_2d.task == task])):
                mask_col = "known_assay" if model_name == "established_independent_2d" else "known_assay"
                known = group.loc[group[mask_col].astype(bool)]
                metrics.append({"seed": seed, "task": task, "model": model_name,
                                "known_assay_n": int(len(known)),
                                "known_assay_coverage": float(group[mask_col].astype(bool).mean()),
                                "known_assay_rmse": float(np.sqrt(mean_squared_error(known.y_true, known.y_pred))),
                                "known_assay_mae": float(mean_absolute_error(known.y_true, known.y_pred)),
                                "known_assay_r2": float(r2_score(known.y_true, known.y_pred)),
                                "scaffolds": int(known.scaffold.nunique())})

    metric_frame = pd.DataFrame(metrics)
    compare_frame = pd.DataFrame(comparisons)
    metric_frame.to_csv(out / "model_metrics.csv", index=False)
    compare_frame.to_csv(out / "paired_scaffold_bootstrap.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "inner_model_selections.csv", index=False)
    pd.DataFrame(validations).to_csv(out / "gate_checks.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out / "paired_oof_predictions.csv", index=False)
    gate = {
        "status": "pass" if gate_pass else "fail",
        "rule": "Stage 12 2D+Uni-Mol2 distribution model must beat both matched nested 2D and established independent 2D on both cell lines and both fresh scaffold seeds",
        "all_eight_point_estimate_comparisons_pass": bool(gate_pass),
        "all_bootstrap_intervals_exclude_zero": bool((compare_frame.scaffold_bootstrap_ci95_high < 0).all()),
        "n_bootstrap": int(args.n_bootstrap),
        "scope": "known-assay internal ChEMBL scaffold CV; no prospective biological validation",
    }
    (out / "stage_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(metric_frame.to_string(index=False))
    print(compare_frame.to_string(index=False))
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
