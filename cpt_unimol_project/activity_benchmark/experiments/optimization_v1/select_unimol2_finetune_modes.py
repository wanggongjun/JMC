#!/usr/bin/env python3
"""Select Uni-Mol2 fine-tuning depth from inner validation only; audit outer OOF afterward."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("hepg2_pIC50", "hct116_pIC50")
MODE_NAMES = ("head", "last2", "full")


def score_rows(table: pd.DataFrame, seed: int) -> dict:
    y = table.y_true.to_numpy(dtype=float)
    pred = table.y_pred.to_numpy(dtype=float)
    groups = table.scaffold.astype(str).to_numpy()
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(5000, dtype=float)
    for i in range(len(deltas)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[group] for group in chosen])
        deltas[i] = np.sqrt(mean_squared_error(y[idx], pred[idx]))
    return {
        "n": int(len(table)), "n_scaffolds": int(len(unique)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred)),
        "scaffold_bootstrap_rmse_ci95_low": float(np.quantile(deltas, 0.025)),
        "scaffold_bootstrap_rmse_ci95_high": float(np.quantile(deltas, 0.975)),
    }


def paired_delta(candidate: pd.DataFrame, baseline: pd.DataFrame, seed: int) -> dict:
    left = candidate.set_index("smiles")
    right = baseline.set_index("smiles")
    joined = left[["scaffold", "y_true", "y_pred"]].join(
        right[["y_true", "y_pred"]].rename(columns={"y_true": "baseline_y",
                                                       "y_pred": "baseline_pred"}),
        how="inner", validate="one_to_one")
    if len(joined) != len(left) or len(joined) != len(right):
        raise RuntimeError("Outer OOF molecule sets do not match exactly")
    if not np.allclose(joined.y_true, joined.baseline_y, atol=1e-6, rtol=0):
        raise RuntimeError("Outer OOF labels differ between the paired models")
    y = joined.y_true.to_numpy(dtype=float)
    p3 = joined.y_pred.to_numpy(dtype=float)
    p2 = joined.baseline_pred.to_numpy(dtype=float)
    groups = joined.scaffold.astype(str).to_numpy()
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(5000, dtype=float)
    for i in range(len(deltas)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[group] for group in chosen])
        deltas[i] = (np.sqrt(mean_squared_error(y[idx], p3[idx]))
                     - np.sqrt(mean_squared_error(y[idx], p2[idx])))
    return {
        "n_paired": int(len(joined)), "n_scaffolds": int(len(unique)),
        "candidate_minus_baseline_rmse": float(
            np.sqrt(mean_squared_error(y, p3)) - np.sqrt(mean_squared_error(y, p2))),
        "paired_scaffold_bootstrap_ci95_low": float(np.quantile(deltas, 0.025)),
        "paired_scaffold_bootstrap_ci95_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_probability_candidate_better": float(np.mean(deltas < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dir", required=True)
    parser.add_argument("--last2-dir", required=True)
    parser.add_argument("--full-dir", required=True)
    parser.add_argument("--matched-2d-dir", required=True)
    parser.add_argument("--full-2d-file", default="experiments/optimization_v1/nested_cv/outer_oof_predictions.csv")
    parser.add_argument("--full-2d-model", default="2d_best")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    roots = {"head": Path(args.head_dir), "last2": Path(args.last2_dir),
             "full": Path(args.full_dir)}
    roots = {key: root if root.is_absolute() else ROOT / root for key, root in roots.items()}
    metrics_by_mode, predictions_by_mode = {}, {}
    for mode, root in roots.items():
        metrics_by_mode[mode] = pd.read_csv(root / "outer_fold_metrics.csv")
        predictions_by_mode[mode] = pd.read_csv(root / "outer_oof_predictions.csv")
    matched_root = Path(args.matched_2d_dir)
    if not matched_root.is_absolute():
        matched_root = ROOT / matched_root
    matched_2d = pd.read_csv(matched_root / "outer_oof_predictions.csv")
    full_path = Path(args.full_2d_file)
    if not full_path.is_absolute():
        full_path = ROOT / full_path
    full_2d = pd.read_csv(full_path)
    full_2d = full_2d.loc[full_2d.model.astype(str) == args.full_2d_model].copy()
    if full_2d.empty:
        raise RuntimeError(f"No rows for full-training 2D model {args.full_2d_model} in {full_path}")

    selections, selected_parts = [], []
    for task in TASKS:
        folds = sorted(set(metrics_by_mode["last2"].loc[
            metrics_by_mode["last2"].task == task, "outer_fold"].astype(int)))
        if len(folds) != 5:
            raise RuntimeError(f"Expected five outer folds for {task}, got {folds}")
        for fold in folds:
            scores = []
            for mode in MODE_NAMES:
                row = metrics_by_mode[mode].loc[
                    (metrics_by_mode[mode].task == task)
                    & (metrics_by_mode[mode].outer_fold.astype(int) == fold)]
                if len(row) != 1:
                    raise RuntimeError(f"Missing/duplicate mode metric: {task}/{fold}/{mode}")
                scores.append((float(row.iloc[0].validation_rmse), mode))
            val_rmse, selected_mode = min(scores, key=lambda item: (item[0], MODE_NAMES.index(item[1])))
            pred = predictions_by_mode[selected_mode]
            pred = pred.loc[(pred.task == task) & (pred.outer_fold.astype(int) == fold)].copy()
            if pred.empty:
                raise RuntimeError(f"No outer predictions for selected mode: {task}/{fold}/{selected_mode}")
            pred["selected_mode"] = selected_mode
            selected_parts.append(pred)
            selections.append({"task": task, "outer_fold": fold,
                               "selected_mode": selected_mode,
                               "inner_validation_rmse": val_rmse,
                               "selection_basis": "inner validation only"})
    chosen = pd.concat(selected_parts, ignore_index=True)
    selection_table = pd.DataFrame(selections)
    chosen.to_csv(output / "selected_unimol2_oof_predictions.csv", index=False)
    selection_table.to_csv(output / "inner_selected_mode_by_fold.csv", index=False)

    summaries, folds_out = [], []
    for task in TASKS:
        task_chosen = chosen.loc[chosen.task == task].copy()
        summary = {"task": task, "model": "unimol2_inner_selected_finetune"}
        summary.update(score_rows(task_chosen, seed=20261000 + TASKS.index(task)))
        for name, base in (("matched_2d", matched_2d), ("full_training_2d", full_2d)):
            paired = base.loc[base.task == task].copy()
            if name == "full_training_2d":
                paired = paired[["task", "outer_fold", "smiles", "scaffold", "y_true", "y_pred"]]
            summary.update({f"{name}_{key}": val for key, val in
                            paired_delta(task_chosen, paired, seed=20261010 + 13 * TASKS.index(task) + (0 if name == "matched_2d" else 1)).items()})
        summaries.append(summary)
        for fold in sorted(task_chosen.outer_fold.astype(int).unique()):
            c = task_chosen.loc[task_chosen.outer_fold.astype(int) == fold]
            m = matched_2d.loc[(matched_2d.task == task)
                               & (matched_2d.outer_fold.astype(int) == fold)]
            folds_out.append({"task": task, "outer_fold": fold,
                              "selected_mode": selection_table.loc[
                                  (selection_table.task == task)
                                  & (selection_table.outer_fold == fold), "selected_mode"].iloc[0],
                              "unimol2_rmse": float(np.sqrt(mean_squared_error(c.y_true, c.y_pred))),
                              "matched_2d_rmse": float(np.sqrt(mean_squared_error(m.y_true, m.y_pred)))})
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(folds_out).to_csv(output / "fold_metrics.csv", index=False)
    audit = {
        "selection_uses_outer_test": False,
        "mode_selection": "minimum inner-validation RMSE independently per task and outer fold",
        "outer_fold_count_per_task": 5,
        "selected_oof_rows": {task: int((chosen.task == task).sum()) for task in TASKS},
        "matched_comparator_fit_fraction": "same 12% scaffold inner validation; same remaining inner training rows",
        "full_training_2d_reference": str(full_path),
        "full_training_2d_model": args.full_2d_model,
        "mode_selection_counts": {
            f"{task}/{mode}": int(count)
            for (task, mode), count in selection_table.groupby(["task", "selected_mode"]).size().items()
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
