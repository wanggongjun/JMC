from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]


def load_predict_fn(path):
    spec = importlib.util.spec_from_file_location("pred_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict_dual


def main(smiles):
    methods = {
        "01_scaffold_stacking_ensemble": ROOT / "campt_activity_models" / "01_scaffold_stacking_ensemble" / "predict.py",
        "02_masked_multitask_mlp": ROOT / "campt_activity_models" / "02_masked_multitask_mlp" / "predict.py",
        "03_tanimoto_krr_conformal": ROOT / "campt_activity_models" / "03_tanimoto_krr_conformal" / "predict.py",
    }
    out = {}
    for name, p in methods.items():
        fn = load_predict_fn(p)
        out[name] = fn(smiles)
    return out


if __name__ == "__main__":
    example = "CC1=C2C(=O)OC3(CC)C(=O)OCC3C2=NC4=CC=CC=C14"
    print(main(example))
