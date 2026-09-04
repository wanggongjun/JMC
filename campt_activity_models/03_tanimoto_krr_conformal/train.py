from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.kernel_ridge import KernelRidge

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "campt_activity_models"))
from common.pipeline_utils import load_task_data, scaffold_split, regression_metrics, save_json, set_seed
from rdkit import Chem
from rdkit.Chem import AllChem


def fps_from_smiles(smiles_list, n_bits=2048, radius=2):
    fps = []
    valid_idx = []
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.fromiter((int(x) for x in fp.ToBitString()), dtype=np.uint8)
        fps.append(arr)
        valid_idx.append(i)
    if not fps:
        return np.zeros((0, n_bits), dtype=np.uint8), np.array(valid_idx, dtype=int)
    return np.stack(fps, axis=0), np.array(valid_idx, dtype=int)


def tanimoto_kernel(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    dot = A @ B.T
    A_sum = A.sum(axis=1, keepdims=True)
    B_sum = B.sum(axis=1, keepdims=True).T
    denom = A_sum + B_sum - dot
    return dot / np.clip(denom, 1e-8, None)


def fit_task(task_name, df, out_dir, seed=42):
    split = scaffold_split(df, seed=seed)

    tr = split["train"].reset_index(drop=True)
    va = split["val"].reset_index(drop=True)
    te = split["test"].reset_index(drop=True)

    Xtr, idx_tr = fps_from_smiles(tr["smiles"].tolist())
    ytr = tr.iloc[idx_tr]["y"].values.astype(np.float32)

    Xva, idx_va = fps_from_smiles(va["smiles"].tolist())
    yva = va.iloc[idx_va]["y"].values.astype(np.float32)

    Xte, idx_te = fps_from_smiles(te["smiles"].tolist())
    yte = te.iloc[idx_te]["y"].values.astype(np.float32)

    Ktr = tanimoto_kernel(Xtr, Xtr)
    model = KernelRidge(alpha=0.8, kernel="precomputed")
    model.fit(Ktr, ytr)

    Kva = tanimoto_kernel(Xva, Xtr)
    pva = model.predict(Kva)
    q90 = float(np.quantile(np.abs(yva - pva), 0.9))

    Kte = tanimoto_kernel(Xte, Xtr)
    pte = model.predict(Kte)
    metrics = regression_metrics(yte, pte)

    pred_df = te.iloc[idx_te].copy()
    pred_df["y_pred"] = pte
    pred_df["pred_lower_90"] = pte - q90
    pred_df["pred_upper_90"] = pte + q90
    pred_df.to_csv(out_dir / f"{task_name}_test_predictions.csv", index=False, encoding="utf-8-sig")

    np.save(out_dir / f"{task_name}_train_fp.npy", Xtr)
    np.save(out_dir / f"{task_name}_train_y.npy", ytr)
    joblib.dump(model, out_dir / f"{task_name}_model.joblib")
    save_json({"q90": q90}, out_dir / f"{task_name}_conformal.json")

    return {
        "task": task_name,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
        "conformal_q90": q90,
        "metrics": metrics,
    }


def main():
    set_seed(42)
    out_dir = ROOT / "campt_activity_models" / "03_tanimoto_krr_conformal" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_task_data(ROOT)
    report = {}
    for name, df in tasks.items():
        report[name] = fit_task(name, df, out_dir, seed=42)

    save_json(report, out_dir / "metrics.json")
    print("saved to", out_dir)
    print(report)


if __name__ == "__main__":
    main()
