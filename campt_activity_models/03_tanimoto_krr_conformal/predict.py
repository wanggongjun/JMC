from pathlib import Path
import sys
import json
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]


def fp_one(smiles, n_bits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.fromiter((int(x) for x in fp.ToBitString()), dtype=np.uint8)


def tanimoto_kernel(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    dot = A @ B.T
    A_sum = A.sum(axis=1, keepdims=True)
    B_sum = B.sum(axis=1, keepdims=True).T
    denom = A_sum + B_sum - dot
    return dot / np.clip(denom, 1e-8, None)


def predict_task(task, fp):
    art = ROOT / "campt_activity_models" / "03_tanimoto_krr_conformal" / "artifacts"
    model = joblib.load(art / f"{task}_model.joblib")
    Xtr = np.load(art / f"{task}_train_fp.npy")
    with open(art / f"{task}_conformal.json", "r", encoding="utf-8") as f:
        q = json.load(f)["q90"]
    K = tanimoto_kernel(fp.reshape(1, -1), Xtr)
    p = float(model.predict(K)[0])
    return p, float(p - q), float(p + q)


def predict_dual(smiles: str):
    fp = fp_one(smiles)
    if fp is None:
        raise ValueError("Invalid SMILES")
    hep = predict_task("hepg2", fp)
    hct = predict_task("hct116", fp)
    return {
        "hepg2_pIC50": hep[0],
        "hepg2_pi90": [hep[1], hep[2]],
        "hct116_pIC50": hct[0],
        "hct116_pi90": [hct[1], hct[2]],
    }


if __name__ == "__main__":
    test_smiles = "CC1=C2C(=O)OC3(CC)C(=O)OCC3C2=NC4=CC=CC=C14"
    print(predict_dual(test_smiles))
