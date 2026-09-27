from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "modeling_dataset.csv"
REPR = ROOT / "results" / "unimol_v1_cls_repr.npz"
OUT = ROOT / "results"
TASKS = ("hepg2_pIC50", "hct116_pIC50")
DESC_FUNCS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors, Descriptors.RingCount,
    Descriptors.HeavyAtomCount, Descriptors.FractionCSP3, Descriptors.NumRotatableBonds,
]


def featurize_2d(smiles_list: list[str]) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES survived cleaning: {smiles}")
        bits = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), bits)
        desc = np.asarray([fn(mol) for fn in DESC_FUNCS], dtype=np.float32)
        rows.append(np.concatenate([bits.astype(np.float32), desc]))
    return np.asarray(rows, dtype=np.float32)


def candidate_models(seed: int):
    for leaf in [2, 8]:
        yield f"extra_trees_leaf_{leaf}", ExtraTreesRegressor(
            n_estimators=120, min_samples_leaf=leaf, max_features="sqrt",
            n_jobs=4, random_state=seed,
        )


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n_test": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
    }


def fit_eval(task: str, feature_name: str, X_all: np.ndarray, subset: pd.DataFrame, split: pd.DataFrame, all_smiles: list[str], seed: int):
    subset = subset.merge(split[["smiles", "split", "scaffold"]], on="smiles", how="left", validate="one_to_one")
    if subset.split.isna().any():
        raise RuntimeError(f"Missing split assignment for {task}")
    train = subset.split.eq("train").to_numpy()
    val = subset.split.eq("val").to_numpy()
    test = subset.split.eq("test").to_numpy()
    y = subset[task].to_numpy(dtype=np.float32)
    # Subset rows are ordered as the modeling table; shared X rows have that same order.
    row_ids = {s: i for i, s in enumerate(all_smiles)}
    X = X_all[[row_ids[s] for s in subset.smiles.astype(str)]]
    candidates = list(candidate_models(seed=seed))
    scored = []
    for name, model in candidates:
        model.fit(X[train], y[train])
        val_pred = model.predict(X[val])
        val_rmse = float(np.sqrt(mean_squared_error(y[val], val_pred)))
        if np.isfinite(val_rmse) and np.isfinite(val_pred).all():
            scored.append((val_rmse, name, model))
    if not scored:
        raise RuntimeError(f"No finite validation predictions for {task}/{feature_name}")
    val_rmse, name, _ = min(scored, key=lambda x: x[0])
    model = dict((model_name, candidate) for model_name, candidate in candidates)[name]
    model.fit(X[train | val], y[train | val])
    joblib.dump(model, OUT / f"{task}_{feature_name}_seed{seed}_model.joblib")
    pred = model.predict(X[test])
    if not np.isfinite(pred).all():
        raise RuntimeError(f"Non-finite test predictions for {task}/{feature_name}")
    predictions = subset.loc[test, ["smiles", task, "scaffold"]].copy()
    predictions["y_pred"] = pred
    predictions.to_csv(OUT / "predictions" / f"{task}_{feature_name}_seed{seed}_test_predictions.csv", index=False, encoding="utf-8-sig")
    return {
        "feature_set": feature_name,
        "task": task,
        "seed": seed,
        "model": name,
        "selection_metric": "validation RMSE",
        "validation_rmse": val_rmse,
        "split_method": "Bemis-Murcko scaffold holdout",
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        **get_metrics(y[test], pred),
    }, set(subset.loc[test, "smiles"].astype(str))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    split_df = pd.read_csv(ROOT / "results" / "repeated_scaffold_splits.csv", encoding="utf-8-sig")
    repr_data = np.load(REPR, allow_pickle=False)
    X3 = repr_data["cls_repr"].astype(np.float32)
    X3_smiles = repr_data["smiles"].astype(str).tolist()
    all_smiles = data.smiles.astype(str).tolist()
    if X3.shape[0] != len(data) or X3_smiles != all_smiles:
        raise RuntimeError("Uni-Mol representation rows do not align with the modeling dataset")
    X2 = featurize_2d(all_smiles)
    np.savez_compressed(OUT / "2d_morgan_rdkit_features.npz", smiles=np.asarray(all_smiles), features=X2)
    results = []
    test_sets = {}
    seeds = sorted(split_df.seed.unique().astype(int).tolist())
    for seed in seeds:
        seed_tests = {}
        for task in TASKS:
            task_data = data.loc[data[task].notna(), ["smiles", task]].sort_values("smiles").reset_index(drop=True)
            task_split = split_df.loc[(split_df.task == task) & (split_df.seed == seed), ["smiles", "split", "scaffold"]]
            print(f"seed={seed} fitting {task}: 2d_Morgan_RDKit", flush=True)
            result2, test2 = fit_eval(task, "2d_morgan_rdkit", X2, task_data, task_split, all_smiles, seed)
            print(f"seed={seed} fitting {task}: 3d_Uni-Mol_v1", flush=True)
            result3, test3 = fit_eval(task, "3d_unimol_v1", X3, task_data, task_split, all_smiles, seed)
            if test2 != test3:
                raise RuntimeError(f"2D/3D test molecules differ for {task}/seed={seed}")
            results.extend([result2, result3])
            seed_tests[task] = len(test2)
        test_sets[str(seed)] = seed_tests
    (OUT / "matched_model_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (OUT / "matched_test_set_audit.json").write_text(json.dumps({"same_test_smiles_for_2d_and_3d_within_each_seed": True, "seeds": seeds, "n_test_per_task_by_seed": test_sets}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
