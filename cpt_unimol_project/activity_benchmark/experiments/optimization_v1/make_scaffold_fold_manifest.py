"""Create a reproducible repeated Bemis-Murcko GroupKFold manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/modeling_dataset.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    rows = []
    for task in ("hepg2_pIC50", "hct116_pIC50"):
        part = data.loc[data[task].notna(), ["smiles", task]].reset_index(drop=True)
        part["scaffold"] = part.smiles.astype(str).map(scaffold)
        groups = part.scaffold.astype(str).to_numpy()
        splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold = [-1] * len(part)
        for fold_i, (_, test_idx) in enumerate(splitter.split(part.smiles, part[task], groups)):
            for row_i in test_idx:
                fold[int(row_i)] = int(fold_i)
        if min(fold) < 0:
            raise RuntimeError(f"Incomplete fold assignment for {task}")
        if any(len(set(groups[np.asarray(fold) == i]) & set(groups[np.asarray(fold) != i]))
               for i in range(args.folds)):
            raise RuntimeError(f"Scaffold leakage in {task}")
        rows.extend({"task": task, "smiles": str(part.iloc[i].smiles),
                     "scaffold": str(part.iloc[i].scaffold), "outer_fold": fold[i]}
                    for i in range(len(part)))
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    fold_stats = (table.groupby(["task", "outer_fold"])
                  .agg(molecules=("smiles", "size"), scaffolds=("scaffold", "nunique"))
                  .reset_index().to_dict(orient="records"))
    meta = {"seed": args.seed, "folds": args.folds,
            "validation": "shuffled Bemis-Murcko GroupKFold; molecules/scaffolds are grouped",
            "dataset_sha256": sha256(DATA), "manifest_sha256": sha256(out),
            "fold_stats": fold_stats}
    out.with_suffix(out.suffix + ".audit.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
