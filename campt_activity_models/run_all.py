from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
py = ROOT / ".venv" / "Scripts" / "python.exe"

scripts = [
    ROOT / "campt_activity_models" / "01_scaffold_stacking_ensemble" / "train.py",
    ROOT / "campt_activity_models" / "02_masked_multitask_mlp" / "train.py",
    ROOT / "campt_activity_models" / "03_tanimoto_krr_conformal" / "train.py",
]

for s in scripts:
    print("\n=== RUN", s, "===")
    subprocess.check_call([str(py), str(s)], cwd=str(ROOT))

print("\nAll methods finished.")
