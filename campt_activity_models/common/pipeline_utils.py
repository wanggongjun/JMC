import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def load_task_data(base_dir: Path):
    hep = pd.read_csv(base_dir / "alldata" / "hepg2_smiles_pIC50.csv", encoding="utf-8-sig")
    hct = pd.read_csv(base_dir / "alldata" / "hct116_smiles_pIC50.csv", encoding="utf-8-sig")

    hep = hep.rename(columns={"canonical_smiles": "smiles", "pIC50": "y"})[["smiles", "y"]]
    hct = hct.rename(columns={"canonical_smiles": "smiles", "pIC50": "y"})[["smiles", "y"]]

    hep["y"] = pd.to_numeric(hep["y"], errors="coerce")
    hct["y"] = pd.to_numeric(hct["y"], errors="coerce")
    hep = hep.dropna().drop_duplicates()
    hct = hct.dropna().drop_duplicates()

    hep = hep.groupby("smiles", as_index=False)["y"].mean()
    hct = hct.groupby("smiles", as_index=False)["y"].mean()

    return {"hepg2": hep, "hct116": hct}


def make_scaffold(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(df: pd.DataFrame, seed: int = 42, train_frac=0.8, val_frac=0.1):
    buckets = {}
    for idx, s in enumerate(df["smiles"].tolist()):
        scf = make_scaffold(s)
        if scf is None:
            continue
        buckets.setdefault(scf, []).append(idx)

    scf_items = list(buckets.items())
    rng = np.random.RandomState(seed)
    rng.shuffle(scf_items)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in scf_items:
        if len(train_idx) + len(idxs) <= n_train:
            train_idx.extend(idxs)
        elif len(val_idx) + len(idxs) <= n_val:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    used = set(train_idx) | set(val_idx) | set(test_idx)
    leftovers = [i for i in range(n) if i not in used]
    test_idx.extend(leftovers)

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


DESC_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("RingCount", Descriptors.RingCount),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
]


def featurize_smiles(smiles_list, n_bits=2048, radius=2):
    features = []
    valid = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            valid.append(False)
            features.append(None)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.fromiter((int(x) for x in fp.ToBitString()), dtype=np.float32)
        desc = np.array([func(mol) for _, func in DESC_FUNCS], dtype=np.float32)
        x = np.concatenate([arr, desc], axis=0)
        valid.append(True)
        features.append(x)

    valid_idx = [i for i, ok in enumerate(valid) if ok]
    X = np.stack([features[i] for i in valid_idx], axis=0) if valid_idx else np.zeros((0, n_bits + len(DESC_FUNCS)), dtype=np.float32)
    return X, np.array(valid, dtype=bool)


def build_task_matrices(task_df: pd.DataFrame):
    X, valid_mask = featurize_smiles(task_df["smiles"].tolist())
    kept = task_df.iloc[np.where(valid_mask)[0]].reset_index(drop=True)
    y = kept["y"].values.astype(np.float32)
    return X, y, kept


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    if len(y_true) > 1:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    else:
        pearson, spearman = 0.0, 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson, "spearman": spearman}


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
