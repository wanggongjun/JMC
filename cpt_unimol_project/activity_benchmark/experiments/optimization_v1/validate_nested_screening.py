"""Recompute and audit the frozen-representation nested-CV artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "nested_cv"
DATA = ROOT / "data/processed/modeling_dataset.csv"
TASKS = ("hepg2_pIC50", "hct116_pIC50")
MODELS = ("2d_best", "3d_best", "2d_3d_hybrid")


def main():
    data = pd.read_csv(DATA, encoding="utf-8-sig").set_index("smiles")
    pred = pd.read_csv(OUT / "outer_oof_predictions.csv")
    folds = pd.read_csv(OUT / "outer_scaffold_folds.csv")
    saved = pd.read_csv(OUT / "pooled_nested_cv_metrics.csv")
    computed = []
    for task in TASKS:
        expected = set(data.index[data[task].notna()].astype(str))
        fp = folds.loc[folds.task == task]
        if fp.smiles.duplicated().any() or set(fp.smiles.astype(str)) != expected:
            raise RuntimeError(f"Outer fold assignments do not cover {task} exactly once")
        if fp.groupby("scaffold").outer_fold.nunique().max() != 1:
            raise RuntimeError(f"Outer scaffold appears in multiple folds for {task}")
        task_pred = pred.loc[pred.task == task]
        for model in MODELS:
            part = task_pred.loc[task_pred.model == model]
            if set(part.smiles.astype(str)) != expected or part.smiles.duplicated().any():
                raise RuntimeError(f"OOF rows do not cover {task}/{model} exactly once")
            truth = data.loc[part.smiles.astype(str), task].to_numpy(dtype=float)
            if not np.allclose(truth, part.y_true.to_numpy(dtype=float), rtol=0, atol=1e-6):
                raise RuntimeError(f"Prediction truth labels disagree with source for {task}/{model}")
            yhat = part.y_pred.to_numpy(dtype=float)
            if not np.isfinite(yhat).all():
                raise RuntimeError(f"Non-finite OOF predictions for {task}/{model}")
            computed.append({"task": task, "model": model,
                             "n": len(part),
                             "rmse": float(np.sqrt(mean_squared_error(truth, yhat))),
                             "mae": float(mean_absolute_error(truth, yhat)),
                             "r2": float(r2_score(truth, yhat))})
    cmp = pd.DataFrame(computed).merge(saved, on=["task", "model"], suffixes=("_recomputed", "_saved"), validate="one_to_one")
    for key in ("n", "rmse", "mae", "r2"):
        if not np.allclose(cmp[f"{key}_recomputed"], cmp[f"{key}_saved"], rtol=0, atol=1e-6):
            raise RuntimeError(f"Saved nested CV {key} values do not match predictions")
    audit = {"status": "passed", "outer_split_method": "5-fold Bemis-Murcko GroupKFold",
             "inner_selection_method": "4-fold scaffold GroupKFold",
             "tasks": list(TASKS), "models": list(MODELS),
             "outer_scaffold_overlap": 0, "complete_paired_oof_rows": True,
             "recomputed_metrics_match_tolerance": 1e-6}
    (OUT / "validation_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
