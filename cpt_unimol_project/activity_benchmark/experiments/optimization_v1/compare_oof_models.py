"""Audit and compare paired outer-OOF predictions for a 3D route vs 2D."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def bootstrap_scaffolds(frame: pd.DataFrame, iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    scaffold_rows = {s: np.flatnonzero(frame.scaffold.to_numpy() == s)
                     for s in frame.scaffold.unique()}
    scaffold_ids = np.asarray(list(scaffold_rows), dtype=object)
    y = frame.y_true.to_numpy(dtype=np.float64)
    y2 = frame.y_pred_2d.to_numpy(dtype=np.float64)
    y3 = frame.y_pred_3d.to_numpy(dtype=np.float64)
    differences = np.empty(iterations, dtype=np.float64)
    for b in range(iterations):
        sampled = rng.choice(scaffold_ids, size=len(scaffold_ids), replace=True)
        ids = np.concatenate([scaffold_rows[s] for s in sampled])
        rmse_2d = np.sqrt(np.mean(np.square(y[ids] - y2[ids])))
        rmse_3d = np.sqrt(np.mean(np.square(y[ids] - y3[ids])))
        differences[b] = rmse_3d - rmse_2d
    return {
        "iterations": int(iterations),
        "seed": int(seed),
        "scaffold_count": int(len(scaffold_ids)),
        "rmse_delta_3d_minus_2d_ci95_low": float(np.quantile(differences, 0.025)),
        "rmse_delta_3d_minus_2d_ci95_high": float(np.quantile(differences, 0.975)),
        "bootstrap_probability_3d_lower_rmse": float(np.mean(differences < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-predictions", required=True,
                        help="CSV with task/outer_fold/smiles/scaffold/y_true/model/y_pred")
    parser.add_argument("--baseline-predictions",
                        default="experiments/optimization_v1/nested_cv/outer_oof_predictions.csv")
    parser.add_argument("--model-name", required=True,
                        help="Model label in --model-predictions, e.g. unimol_v1_finetuned")
    parser.add_argument("--baseline-name", default="2d_best")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Smoke checks only; do not use partial outputs as research results")
    args = parser.parse_args()
    if args.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be positive")

    model_path = Path(args.model_predictions)
    baseline_path = Path(args.baseline_predictions)
    out_dir = Path(args.output_dir)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    model = pd.read_csv(model_path)
    baseline = pd.read_csv(baseline_path)
    model = model.loc[model.model == args.model_name].copy()
    baseline = baseline.loc[baseline.model == args.baseline_name].copy()
    key = ["task", "outer_fold", "smiles"]
    if model.duplicated(key).any() or baseline.duplicated(key).any():
        raise RuntimeError("Duplicate predictions found for task/fold/SMILES")
    paired = model.merge(baseline, on=key, suffixes=("_3d", "_2d"), validate="one_to_one")
    if paired.empty:
        raise RuntimeError("No paired predictions; check model labels and scaffold assignments")
    if not np.allclose(paired.y_true_3d, paired.y_true_2d, atol=1e-7, rtol=0):
        raise RuntimeError("Paired model files disagree on ground-truth labels")
    if not np.array_equal(paired.scaffold_3d.astype(str), paired.scaffold_2d.astype(str)):
        raise RuntimeError("Paired model files disagree on scaffold assignments")
    paired = paired.rename(columns={"y_true_3d": "y_true", "y_pred_3d": "y_pred_3d",
                                    "scaffold_3d": "scaffold", "y_pred_2d": "y_pred_2d"})

    required_tasks = {"hepg2_pIC50", "hct116_pIC50"}
    evaluated_tasks = sorted(required_tasks & set(paired.task)) if args.allow_partial else sorted(required_tasks)
    summary_rows, fold_rows, bootstrap_rows = [], [], []
    for task in evaluated_tasks:
        part = paired.loc[paired.task == task].copy()
        folds = sorted(part.outer_fold.unique().tolist())
        if not args.allow_partial and len(folds) != args.expected_folds:
            raise RuntimeError(f"{task}: expected {args.expected_folds} paired outer folds, found {folds}")
        if not args.allow_partial and len(model.loc[model.task == task]) != len(baseline.loc[baseline.task == task]):
            raise RuntimeError(f"{task}: prediction counts differ between model and baseline")
        if not args.allow_partial and len(part) != len(model.loc[model.task == task]):
            raise RuntimeError(f"{task}: paired prediction coverage is incomplete")
        m2 = metrics(part.y_true.to_numpy(), part.y_pred_2d.to_numpy())
        m3 = metrics(part.y_true.to_numpy(), part.y_pred_3d.to_numpy())
        bootstrap = bootstrap_scaffolds(part, args.bootstrap_iterations, args.seed)
        summary_rows.append({
            "task": task,
            "model": args.model_name,
            "baseline": args.baseline_name,
            "n_paired": len(part),
            "outer_folds": len(folds),
            "rmse_2d_pooled": m2["rmse"],
            "rmse_3d_pooled": m3["rmse"],
            "rmse_improvement_3d_minus_2d": m3["rmse"] - m2["rmse"],
            "rmse_relative_improvement_pct": 100 * (m2["rmse"] - m3["rmse"]) / m2["rmse"],
            "mae_2d": m2["mae"], "mae_3d": m3["mae"],
            "r2_2d": m2["r2"], "r2_3d": m3["r2"],
            "folds_3d_win": 0,
            **bootstrap,
        })
        for fold in folds:
            fp = part.loc[part.outer_fold == fold]
            f2 = metrics(fp.y_true.to_numpy(), fp.y_pred_2d.to_numpy())
            f3 = metrics(fp.y_true.to_numpy(), fp.y_pred_3d.to_numpy())
            fold_rows.append({"task": task, "outer_fold": int(fold),
                              "n": len(fp), "rmse_2d": f2["rmse"], "rmse_3d": f3["rmse"],
                              "rmse_delta_3d_minus_2d": f3["rmse"] - f2["rmse"]})
        summary_rows[-1]["folds_3d_win"] = sum(
            row["task"] == task and row["rmse_3d"] < row["rmse_2d"] for row in fold_rows
        )
        bootstrap_rows.append({"task": task, **bootstrap})

    if not args.allow_partial:
        for task in required_tasks:
            if task not in set(paired.task):
                raise RuntimeError(f"Missing paired predictions for required task {task}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_dir / "paired_oof_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "paired_fold_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(out_dir / "scaffold_bootstrap_ci.csv", index=False)
    audit = {
        "status": "smoke_only" if args.allow_partial else "complete",
        "model_predictions": str(model_path),
        "baseline_predictions": str(baseline_path),
        "model_name": args.model_name,
        "baseline_name": args.baseline_name,
        "paired_rows": len(paired),
        "tasks": sorted(paired.task.unique().tolist()),
        "expected_fold_count": args.expected_folds,
        "bootstrap_unit": "Bemis-Murcko scaffold cluster",
        "warning": "Internal ChEMBL scaffold CV is not external prospective validation.",
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
