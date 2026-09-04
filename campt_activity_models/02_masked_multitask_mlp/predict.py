from pathlib import Path
import sys
import joblib
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "campt_activity_models"))
from common.pipeline_utils import featurize_smiles


class MaskedMTL(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head_hep = nn.Linear(256, 1)
        self.head_hct = nn.Linear(256, 1)

    def forward(self, x):
        h = self.backbone(x)
        return torch.cat([self.head_hep(h), self.head_hct(h)], dim=1)


def load_bundle(device):
    art = ROOT / "campt_activity_models" / "02_masked_multitask_mlp" / "artifacts"
    scaler = joblib.load(art / "scaler.joblib")
    in_dim = scaler.mean_.shape[0]
    model = MaskedMTL(in_dim)
    model.load_state_dict(torch.load(art / "mtl_model.pt", map_location=device))
    model.to(device).eval()
    return model, scaler


def predict_dual(smiles: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, valid = featurize_smiles([smiles])
    if not valid[0]:
        raise ValueError("Invalid SMILES")
    model, scaler = load_bundle(device)
    X = scaler.transform(X)
    with torch.no_grad():
        p = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()[0]
    return {"hepg2_pIC50": float(p[0]), "hct116_pIC50": float(p[1])}


if __name__ == "__main__":
    test_smiles = "CC1=C2C(=O)OC3(CC)C(=O)OCC3C2=NC4=CC=CC=C14"
    print(predict_dual(test_smiles))
