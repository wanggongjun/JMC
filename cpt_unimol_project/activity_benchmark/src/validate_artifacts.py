from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
TASKS = ("hepg2_pIC50", "hct116_pIC50")
FEATURES = ("2d_morgan_rdkit", "3d_unimol_v1")
SEEDS = (42, 43, 44, 45, 46)


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "processed" / "modeling_dataset.csv", encoding="utf-8-sig").set_index("smiles")
    split = pd.read_csv(RESULTS / "repeated_scaffold_splits.csv", encoding="utf-8-sig")
    metrics = json.loads((RESULTS / "matched_model_metrics.json").read_text(encoding="utf-8"))
    split_audit = json.loads((RESULTS / "repeated_split_audit.json").read_text(encoding="utf-8"))
    representation = np.load(RESULTS / "unimol_v1_cls_repr.npz", allow_pickle=False)
    repr_metadata = json.loads((RESULTS / "unimol_v1_cls_repr.metadata.json").read_text(encoding="utf-8"))

    cls = representation["cls_repr"]
    repr_smiles = representation["smiles"].astype(str).tolist()
    if cls.shape != (len(data), 512) or not np.isfinite(cls).all():
        raise RuntimeError(f"Unexpected or non-finite Uni-Mol representation matrix: {cls.shape}")
    if repr_smiles != sorted(data.index.astype(str).tolist()):
        raise RuntimeError("Uni-Mol representation smiles do not align with the modeling dataset")
    if repr_metadata["input_molecules"] != len(data) or repr_metadata["embedding_shape"] != list(cls.shape):
        raise RuntimeError("Uni-Mol representation metadata does not match the saved matrix")
    if tuple(sorted(map(int, split_audit["seeds"]))) != SEEDS:
        raise RuntimeError("Split audit does not contain the declared five seeds")
    if any(value != 0 for audit in split_audit["audits"].values() for task in audit["scaffold_overlap"].values() for value in task.values()):
        raise RuntimeError("Scaffold overlap detected")

    metric_map = {(int(row["seed"]), row["task"], row["feature_set"]): row for row in metrics}
    expected_keys = {(seed, task, feature) for seed in SEEDS for task in TASKS for feature in FEATURES}
    if set(metric_map) != expected_keys:
        raise RuntimeError("Saved metrics do not cover every seed/task/feature combination exactly once")
    model_paths = list(RESULTS.glob("*_seed*_model.joblib"))
    prediction_paths = list(PREDICTIONS.glob("*_seed*_test_predictions.csv"))
    if len(model_paths) != len(expected_keys) or len(prediction_paths) != len(expected_keys):
        raise RuntimeError("Saved model or prediction artifact count is incomplete")

    compared_pairs = 0
    tolerance = 1e-6
    for seed in SEEDS:
        for task in TASKS:
            subset = split[(split["seed"] == seed) & (split["task"] == task)]
            expected_test = set(subset.loc[subset["split"] == "test", "smiles"].astype(str))
            paired_sets = []
            for feature in FEATURES:
                path = PREDICTIONS / f"{task}_{feature}_seed{seed}_test_predictions.csv"
                predictions = pd.read_csv(path, encoding="utf-8-sig")
                molecules = set(predictions["smiles"].astype(str))
                if molecules != expected_test or not np.isfinite(predictions["y_pred"].to_numpy()).all():
                    raise RuntimeError(f"Invalid test predictions: {path.name}")
                truth = data.loc[predictions["smiles"].astype(str), task].to_numpy(dtype=np.float64)
                saved_truth = predictions[task].to_numpy(dtype=np.float64)
                pred = predictions["y_pred"].to_numpy(dtype=np.float64)
                if not np.allclose(truth, saved_truth, rtol=0, atol=tolerance):
                    raise RuntimeError(f"Saved target labels disagree with data: {path.name}")
                recomputed = {
                    "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
                    "mae": float(mean_absolute_error(truth, pred)),
                    "r2": float(r2_score(truth, pred)),
                }
                saved = metric_map[(seed, task, feature)]
                if any(not math.isclose(recomputed[key], float(saved[key]), rel_tol=0, abs_tol=tolerance) for key in recomputed):
                    raise RuntimeError(f"Saved metrics do not match prediction file: {path.name}")
                paired_sets.append(molecules)
            if paired_sets[0] != paired_sets[1]:
                raise RuntimeError(f"2D/3D test samples differ for {task}, seed={seed}")
            compared_pairs += 1

    audit = {
        "status": "passed",
        "n_seeds": len(SEEDS),
        "n_tasks": len(TASKS),
        "n_feature_sets": len(FEATURES),
        "representation_shape": list(cls.shape),
        "metric_rows": len(metrics),
        "model_files": len(model_paths),
        "prediction_files": len(prediction_paths),
        "paired_test_sets_checked": compared_pairs,
        "scaffold_overlap": 0,
        "recomputed_metric_tolerance": tolerance,
    }
    target = RESULTS / "final_artifact_audit.json"
    target.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
