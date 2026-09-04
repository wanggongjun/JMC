from pathlib import Path
import sys
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "campt_activity_models"))
from common.pipeline_utils import load_task_data, scaffold_split, featurize_smiles, regression_metrics, save_json, set_seed


class MultiTaskDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


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
        y1 = self.head_hep(h)
        y2 = self.head_hct(h)
        return torch.cat([y1, y2], dim=1)


def make_union_table(task_splits):
    pools = {}
    for task_name, split in task_splits.items():
        for part in ["train", "val", "test"]:
            df = split[part]
            for _, row in df.iterrows():
                s = row["smiles"]
                pools.setdefault(s, {"smiles": s, "hepg2": np.nan, "hct116": np.nan})
                pools[s][task_name] = row["y"]
    return list(pools.values())


def build_part_arrays(union_rows, part_smiles, scaler=None, fit_scaler=False):
    rows = [r for r in union_rows if r["smiles"] in part_smiles]
    smiles = [r["smiles"] for r in rows]
    X_raw, valid = featurize_smiles(smiles)
    rows = [rows[i] for i, v in enumerate(valid) if v]

    y = np.array([[r["hepg2"], r["hct116"]] for r in rows], dtype=np.float32)
    mask = (~np.isnan(y)).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        X = scaler.transform(X_raw)

    return X.astype(np.float32), y, mask, scaler, rows


def evaluate_task(model, scaler, df_task, task_idx, device):
    smiles = df_task["smiles"].tolist()
    y_true = df_task["y"].values.astype(np.float32)
    X, valid = featurize_smiles(smiles)
    y_true = y_true[valid]
    X = scaler.transform(X)
    with torch.no_grad():
        yp = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()[:, task_idx]
    return regression_metrics(y_true, yp)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = ROOT / "campt_activity_models" / "02_masked_multitask_mlp" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_task_data(ROOT)
    splits = {k: scaffold_split(v, seed=42) for k, v in tasks.items()}

    union_rows = make_union_table(splits)
    train_smiles = set(splits["hepg2"]["train"]["smiles"]).union(set(splits["hct116"]["train"]["smiles"]))
    val_smiles = set(splits["hepg2"]["val"]["smiles"]).union(set(splits["hct116"]["val"]["smiles"]))

    X_train, y_train, m_train, scaler, _ = build_part_arrays(union_rows, train_smiles, fit_scaler=True)
    X_val, y_val, m_val, _, _ = build_part_arrays(union_rows, val_smiles, scaler=scaler, fit_scaler=False)

    train_ds = MultiTaskDataset(X_train, y_train, m_train)
    val_ds = MultiTaskDataset(X_val, y_val, m_val)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = MaskedMTL(in_dim=X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val = 1e9
    best_state = None
    patience, wait = 20, 0

    for epoch in range(200):
        model.train()
        for xb, yb, mb in train_dl:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            pred = model(xb)
            loss = (((pred - yb) ** 2) * mb).sum() / (mb.sum() + 1e-8)
            noise = torch.randn_like(xb) * 0.01
            pred_aug = model(xb + noise)
            loss = loss + 0.1 * (((pred_aug - yb) ** 2) * mb).sum() / (mb.sum() + 1e-8)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        val_w = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_dl:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb)
                l = (((pred - yb) ** 2) * mb).sum()
                val_loss += float(l.item())
                val_w += float(mb.sum().item())
        val_loss = val_loss / (val_w + 1e-8)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    hepg2_metrics = evaluate_task(model, scaler, splits["hepg2"]["test"], 0, device)
    hct116_metrics = evaluate_task(model, scaler, splits["hct116"]["test"], 1, device)

    torch.save(model.state_dict(), out_dir / "mtl_model.pt")
    joblib.dump(scaler, out_dir / "scaler.joblib")

    report = {
        "hepg2": {
            "n_train": int(len(splits["hepg2"]["train"])),
            "n_val": int(len(splits["hepg2"]["val"])),
            "n_test": int(len(splits["hepg2"]["test"])),
            "metrics": hepg2_metrics,
        },
        "hct116": {
            "n_train": int(len(splits["hct116"]["train"])),
            "n_val": int(len(splits["hct116"]["val"])),
            "n_test": int(len(splits["hct116"]["test"])),
            "metrics": hct116_metrics,
        },
        "best_val_masked_mse": float(best_val),
    }

    save_json(report, out_dir / "metrics.json")
    print("saved to", out_dir)
    print(report)


if __name__ == "__main__":
    main()
