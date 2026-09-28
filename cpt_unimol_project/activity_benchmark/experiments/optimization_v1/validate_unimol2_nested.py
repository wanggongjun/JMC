"""Independently audit the completed Uni-Mol2 nested scaffold-CV artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "nested_unimol2_84m_cv"
DATA = ROOT / "data/processed/modeling_dataset.csv"
BASELINE = Path(__file__).resolve().parent / "nested_cv"
TASKS = ("hepg2_pIC50", "hct116_pIC50")
MODELS = ("2d_best", "3d_best", "2d_3d_early_fusion", "2d_3d_hybrid")


def main() -> None:
    data = pd.read_csv(DATA, encoding="utf-8-sig").set_index("smiles")
    predictions = pd.read_csv(OUT / "outer_oof_predictions.csv")
    folds = pd.read_csv(OUT / "outer_scaffold_folds.csv")
    baseline_folds = pd.read_csv(BASELINE / "outer_scaffold_folds.csv")
    saved = pd.read_csv(OUT / "pooled_nested_cv_metrics.csv")
    recomputed = []
    paired_rows = 0

    for task in TASKS:
        expected = set(data.index[data[task].notna()].astype(str))
        assignment = folds.loc[folds.task == task].copy()
        assignment["scaffold"] = assignment["scaffold"].fillna("").astype(str)
        if assignment.smiles.duplicated().any() or set(assignment.smiles.astype(str)) != expected:
            raise RuntimeError(f"Outer folds do not cover {task} exactly once")
        if assignment.groupby("scaffold").outer_fold.nunique().max() != 1:
            raise RuntimeError(f"A Bemis-Murcko scaffold crosses folds for {task}")

        reference = baseline_folds.loc[baseline_folds.task == task, ["smiles", "outer_fold", "scaffold"]]
        reference["scaffold"] = reference["scaffold"].fillna("").astype(str)
        check = assignment[["smiles", "outer_fold", "scaffold"]].merge(
            reference, on="smiles", suffixes=("_model", "_baseline"), validate="one_to_one")
        if len(check) != len(expected) or not np.array_equal(check.outer_fold_model, check.outer_fold_baseline):
            raise RuntimeError(f"Uni-Mol2 outer folds do not match the frozen 2D baseline for {task}")
        if not np.array_equal(check.scaffold_model.astype(str), check.scaffold_baseline.astype(str)):
            raise RuntimeError(f"Scaffold labels differ from the frozen 2D baseline for {task}")

        task_predictions = predictions.loc[predictions.task == task]
        if set(task_predictions.model) != set(MODELS):
            raise RuntimeError(f"Unexpected model variants for {task}: {sorted(task_predictions.model.unique())}")
        per_molecule = task_predictions.groupby("smiles").model.nunique()
        if set(per_molecule.index.astype(str)) != expected or not (per_molecule == len(MODELS)).all():
            raise RuntimeError(f"Incomplete/duplicate model prediction coverage for {task}")

        base = task_predictions.loc[task_predictions.model == "2d_best"].set_index("smiles")
        for model in MODELS:
            part = task_predictions.loc[task_predictions.model == model].copy()
            if part.smiles.duplicated().any() or set(part.smiles.astype(str)) != expected:
                raise RuntimeError(f"OOF predictions do not cover {task}/{model} once")
            truth = data.loc[part.smiles.astype(str), task].to_numpy(dtype=float)
            if not np.allclose(truth, part.y_true.to_numpy(dtype=float), rtol=0, atol=1e-6):
                raise RuntimeError(f"Source labels do not match {task}/{model}")
            if not np.isfinite(part.y_pred.to_numpy(dtype=float)).all():
                raise RuntimeError(f"Non-finite predictions in {task}/{model}")
            matched = part.set_index("smiles").join(base[["outer_fold", "scaffold", "y_true"]],
                                                      rsuffix="_2d", validate="one_to_one")
            if not np.array_equal(matched.outer_fold, matched.outer_fold_2d):
                raise RuntimeError(f"OOF folds are not paired for {task}/{model}")
            if not np.allclose(matched.y_true, matched.y_true_2d, atol=1e-7, rtol=0):
                raise RuntimeError(f"Paired labels differ for {task}/{model}")
            paired_rows += len(part)
            y = part.y_true.to_numpy(dtype=float)
            yhat = part.y_pred.to_numpy(dtype=float)
            recomputed.append({"task": task, "model": model, "n": len(part),
                               "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
                               "mae": float(mean_absolute_error(y, yhat)),
                               "r2": float(r2_score(y, yhat))})

    actual = pd.DataFrame(recomputed).merge(saved, on=["task", "model"],
                                            suffixes=("_recomputed", "_saved"), validate="one_to_one")
    for metric in ("n", "rmse", "mae", "r2"):
        if not np.allclose(actual[f"{metric}_recomputed"], actual[f"{metric}_saved"], rtol=0, atol=1e-6):
            raise RuntimeError(f"Saved pooled {metric} does not match OOF predictions")

    audit = {
        "status": "passed",
        "tasks": list(TASKS),
        "models": list(MODELS),
        "outer_protocol": "5-fold Bemis-Murcko GroupKFold, exact fold match to paired 2D baseline",
        "inner_model_selection": "4-fold Bemis-Murcko GroupKFold",
        "outer_scaffold_overlap": 0,
        "paired_model_prediction_rows": paired_rows,
        "source_label_coverage": "complete; maximum tolerated float32 source-label roundoff 1e-6",
        "pooled_metrics_recomputed_tolerance": 1e-6,
        "scope": "Internal nested scaffold CV on the existing ChEMBL cohort; not external or prospective validation.",
    }
    (OUT / "validation_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
