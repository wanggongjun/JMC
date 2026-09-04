from pathlib import Path
import sys
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "campt_activity_models"))
from common.pipeline_utils import featurize_smiles


def load_models():
    art = ROOT / "campt_activity_models" / "01_scaffold_stacking_ensemble" / "artifacts"
    return {
        "hepg2": joblib.load(art / "hepg2_model.joblib"),
        "hct116": joblib.load(art / "hct116_model.joblib"),
    }


def predict_dual(smiles: str):
    X, valid = featurize_smiles([smiles])
    if not valid[0]:
        raise ValueError("Invalid SMILES")
    models = load_models()
    return {
        "hepg2_pIC50": float(models["hepg2"].predict(X)[0]),
        "hct116_pIC50": float(models["hct116"].predict(X)[0]),
    }


if __name__ == "__main__":
    test_smiles = "CC1=C2C(=O)OC3(CC)C(=O)OCC3C2=NC4=CC=CC=C14"
    print(predict_dual(test_smiles))
