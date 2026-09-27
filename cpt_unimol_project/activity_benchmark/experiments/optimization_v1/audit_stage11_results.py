#!/usr/bin/env python3
"""Validate Stage 11 OOF pairing and compare Uni-Mol2 with two 2D baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/optimization_v1/stage11_result_audit"
TASKS = ("hepg2_pIC50", "hct116_pIC50")
SEEDS = ("20261007", "20261008")
MODEL_2D = "joint_task_conditioned_2d"
MODEL_3D = "joint_task_conditioned_2d_plus_unimol2_3d"
MODEL_INDEP_2D = "established_independent_2d"


def bootstrap_pair(candidate: pd.DataFrame, baseline: pd.DataFrame,
                   seed: int, n_boot: int = 5000) -> dict:
    c = candidate.loc[candidate.known_assay].set_index("record_id")
    b = baseline.loc[baseline.known_assay].set_index("record_id")
    joined = b[["scaffold", "y_true", "y_pred"]].join(
        c[["y_true", "y_pred"]].rename(columns={"y_true": "candidate_y", "y_pred": "candidate_pred"}),
        how="inner", validate="one_to_one")
    if len(joined) != len(b) or not np.allclose(joined.y_true, joined.candidate_y, atol=1e-7, rtol=0):
        raise RuntimeError("Candidate and baseline do not share exact known-assay OOF records")
    y = joined.y_true.to_numpy(dtype=np.float64)
    p0 = joined.y_pred.to_numpy(dtype=np.float64)
    p1 = joined.candidate_pred.to_numpy(dtype=np.float64)
    groups = joined.scaffold.astype(str).to_numpy()
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([by_group[group] for group in chosen])
        deltas[i] = (np.sqrt(np.mean((p1[ix] - y[ix]) ** 2))
                     - np.sqrt(np.mean((p0[ix] - y[ix]) ** 2)))
    return {"n_paired_known_assay_observations": int(len(joined)),
            "delta_candidate_minus_2d_rmse": float(np.sqrt(np.mean((p1-y)**2))
                                                    - np.sqrt(np.mean((p0-y)**2))),
            "scaffold_bootstrap_ci95_low": float(np.quantile(deltas, .025)),
            "scaffold_bootstrap_ci95_high": float(np.quantile(deltas, .975)),
            "bootstrap_probability_candidate_better": float(np.mean(deltas < 0)),
            "scaffold_groups": int(len(unique))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--output-dir", default=str(OUT.relative_to(ROOT)))
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    model_metrics = []
    selections = []
    comparisons = []
    validations = []
    all_pass = True

    for seed in SEEDS:
        base_dir = ROOT / "experiments/optimization_v1"
        joint_dir = base_dir / f"multitask_seed_{seed}"
        independent_dir = base_dir / f"independent_2d_seed_{seed}"
        for directory in (joint_dir, independent_dir):
            audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
            if audit.get("status") != "complete":
                raise RuntimeError(f"Incomplete or smoke-only audit: {directory}")
            if audit.get("cross_task_scaffold_leakage") != 0 or audit.get("outer_molecule_overlap") != 0:
                raise RuntimeError(f"Leakage audit failed: {directory}")
        manifest = pd.read_csv(base_dir / f"shared_scaffold_folds_seed_{seed}.csv")
        if manifest.groupby("scaffold").outer_fold.nunique().max() != 1:
            raise RuntimeError(f"Shared scaffold manifest leakage for seed {seed}")
        joint = pd.read_csv(joint_dir / "outer_oof_predictions.csv")
        independent = pd.read_csv(independent_dir / "outer_oof_predictions.csv")
        expected_models = {MODEL_2D, MODEL_3D, "assay_mean_only"}
        if set(joint.model.unique()) != expected_models:
            raise RuntimeError(f"Unexpected joint model set for {seed}: {joint.model.unique()}")
        if set(independent.model.unique()) != {MODEL_INDEP_2D, "assay_mean_only"}:
            raise RuntimeError(f"Unexpected independent 2D model set for {seed}: {independent.model.unique()}")
        pred3 = joint.loc[joint.model == MODEL_3D].copy()
        records3 = set(pred3.record_id.astype(int))
        records_ref = set(independent.loc[independent.model == MODEL_INDEP_2D, "record_id"].astype(int))
        if records3 != records_ref:
            raise RuntimeError(f"OOF record coverage differs for seed {seed}")
        for model in (MODEL_2D, MODEL_3D):
            part = joint.loc[joint.model == model]
            if part.groupby("record_id").size().max() != 1 or not np.isfinite(part.y_pred).all():
                raise RuntimeError(f"Duplicate/non-finite {model} OOF rows for seed {seed}")
            if not part.groupby("record_id").known_assay.nunique().eq(1).all():
                raise RuntimeError(f"Inconsistent known-assay support for {model}/{seed}")
        if not np.isfinite(independent.loc[independent.model == MODEL_INDEP_2D, "y_pred"]).all():
            raise RuntimeError(f"Non-finite independent 2D predictions for seed {seed}")

        selections.extend(pd.read_csv(joint_dir / "inner_model_selections.csv").assign(seed=seed).to_dict("records"))
        for task in TASKS:
            for model_frame, model_name in ((joint, MODEL_2D), (joint, MODEL_3D),
                                            (independent, MODEL_INDEP_2D)):
                p = model_frame.loc[(model_frame.task == task) & (model_frame.model == model_name)]
                known = p.loc[p.known_assay]
                model_metrics.append({"seed": seed, "task": task, "model": model_name,
                                      "n_known_assay": int(len(known)),
                                      "known_assay_coverage": float(p.known_assay.mean()),
                                      "known_assay_rmse": float(np.sqrt(mean_squared_error(known.y_true, known.y_pred))),
                                      "known_assay_mae": float(mean_absolute_error(known.y_true, known.y_pred)),
                                      "known_assay_r2": float(r2_score(known.y_true, known.y_pred)),
                                      "scaffolds": int(known.scaffold.nunique())})
            cand = joint.loc[(joint.task == task) & (joint.model == MODEL_3D)].copy()
            for baseline_frame, baseline_name in ((joint, MODEL_2D),
                                                  (independent, MODEL_INDEP_2D)):
                ref = baseline_frame.loc[(baseline_frame.task == task)
                                         & (baseline_frame.model == baseline_name)].copy()
                result = bootstrap_pair(cand, ref,
                                        seed=20263000 + int(seed) + (0 if task == TASKS[0] else 100)
                                        + (0 if baseline_name == MODEL_2D else 10),
                                        n_boot=args.n_bootstrap)
                row = {"seed": seed, "task": task, "candidate": MODEL_3D,
                       "baseline": baseline_name, **result}
                comparisons.append(row)
                if result["delta_candidate_minus_2d_rmse"] >= 0:
                    all_pass = False
                validations.append({"seed": seed, "task": task, "baseline": baseline_name,
                                    "paired_exact_rows": result["n_paired_known_assay_observations"],
                                    "pass_point_estimate": result["delta_candidate_minus_2d_rmse"] < 0})

    metrics = pd.DataFrame(model_metrics)
    compare = pd.DataFrame(comparisons)
    select = pd.DataFrame(selections)
    metrics.to_csv(out / "model_metrics.csv", index=False)
    compare.to_csv(out / "paired_scaffold_bootstrap.csv", index=False)
    select.to_csv(out / "inner_model_selections.csv", index=False)
    pd.DataFrame(validations).to_csv(out / "gate_checks.csv", index=False)
    gate = {
        "status": "pass" if all_pass else "fail",
        "rule": "2D+Uni-Mol2 point RMSE must be lower than both matched joint 2D and established independent 2D in both tasks on both fresh seeds",
        "n_bootstrap": int(args.n_bootstrap), "all_task_seed_baseline_comparisons_pass": bool(all_pass),
        "all_intervals_exclude_zero": bool(((compare.scaffold_bootstrap_ci95_high < 0)
                                             | (compare.scaffold_bootstrap_ci95_low > 0)).all()),
        "note": "Bootstrap intervals describe scaffold-level uncertainty. This remains internal ChEMBL evaluation, not prospective validation.",
    }
    (out / "stage_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(compare.to_string(index=False))
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
