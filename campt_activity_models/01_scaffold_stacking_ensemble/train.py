from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "campt_activity_models"))
from common.pipeline_utils import load_task_data, scaffold_split, build_task_matrices, regression_metrics, save_json, featurize_smiles, set_seed


def build_model(seed=42):
    base_estimators = [
        ("rf", RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)),
        ("et", ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=-1)),
        ("gbr", GradientBoostingRegressor(random_state=seed, n_estimators=300, learning_rate=0.03, max_depth=3)),
    ]
    final_estimator = RidgeCV(alphas=np.logspace(-3, 3, 13))
    stack = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5,
        passthrough=True,
        n_jobs=-1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("stack", stack),
    ])


def train_task(task_name, df, out_dir, seed=42):
    split = scaffold_split(df, seed=seed)
    X_train, y_train, train_df = build_task_matrices(split["train"])
    X_val, y_val, val_df = build_task_matrices(split["val"])
    X_test, y_test, test_df = build_task_matrices(split["test"])

    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    model = build_model(seed=seed)
    model.fit(X_fit, y_fit)

    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)

    pred_df = test_df.copy()
    pred_df["y_pred"] = pred
    pred_df.to_csv(out_dir / f"{task_name}_test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(model, out_dir / f"{task_name}_model.joblib")

    return {
        "task": task_name,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
    }


def main():
    set_seed(42)
    out_dir = ROOT / "campt_activity_models" / "01_scaffold_stacking_ensemble" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_task_data(ROOT)
    report = {}
    for name, df in tasks.items():
        report[name] = train_task(name, df, out_dir, seed=42)

    save_json(report, out_dir / "metrics.json")
    print("saved to", out_dir)
    print(report)


if __name__ == "__main__":
    main()
