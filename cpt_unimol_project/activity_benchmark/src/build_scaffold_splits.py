from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "modeling_dataset.csv"
OUT = ROOT / "results" / "scaffold_splits.csv"
REPEATED_OUT = ROOT / "results" / "repeated_scaffold_splits.csv"


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES in cleaned dataset: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def split_one(df: pd.DataFrame, task: str, seed: int) -> pd.DataFrame:
    # Mirrors the pre-existing baseline split so the 2D reference metrics are comparable.
    buckets: dict[str, list[int]] = {}
    for idx, smiles in enumerate(df.smiles.tolist()):
        buckets.setdefault(scaffold(smiles), []).append(idx)
    items = list(buckets.items())
    np.random.RandomState(seed).shuffle(items)
    n = len(df)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for _, idxs in items:
        if len(train_idx) + len(idxs) <= n_train:
            train_idx.extend(idxs)
        elif len(val_idx) + len(idxs) <= n_val:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)
    used = set(train_idx) | set(val_idx) | set(test_idx)
    test_idx.extend(i for i in range(n) if i not in used)
    labels = {i: "train" for i in train_idx}
    labels.update({i: "val" for i in val_idx})
    labels.update({i: "test" for i in test_idx})
    out = df[["smiles"]].copy()
    out["task"] = task
    out["scaffold"] = [scaffold(s) for s in out.smiles]
    out["split"] = [labels[i] for i in range(len(out))]
    out["seed"] = seed
    return out


def audit_one(result: pd.DataFrame, seed: int) -> dict:
    counts = result.groupby(["task", "split"]).size().unstack(fill_value=0).to_dict(orient="index")
    overlap = {}
    for task, group in result.groupby("task"):
        scaffold_sets = {name: set(g.scaffold) for name, g in group.groupby("split")}
        overlap[task] = {
            "train_val": len(scaffold_sets["train"] & scaffold_sets["val"]),
            "train_test": len(scaffold_sets["train"] & scaffold_sets["test"]),
            "val_test": len(scaffold_sets["val"] & scaffold_sets["test"]),
        }
    return {
        "seed": seed,
        "split_method": "Bemis-Murcko scaffold grouped holdout",
        "fractions_target": {"train": 0.8, "val": 0.1, "test": 0.1},
        "counts": counts,
        "scaffold_overlap": overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    seeds = args.seeds
    seed_results = {}
    for seed in seeds:
        task_splits = []
        for task in ("hepg2_pIC50", "hct116_pIC50"):
            observed = df.loc[df[task].notna(), ["smiles", task]].sort_values("smiles").reset_index(drop=True)
            task_splits.append(split_one(observed, task, seed))
        seed_results[seed] = pd.concat(task_splits, ignore_index=True)
    repeated = pd.concat(seed_results.values(), ignore_index=True)
    primary_seed = 42 if 42 in seed_results else seeds[0]
    result = seed_results[primary_seed]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    repeated.to_csv(REPEATED_OUT, index=False, encoding="utf-8-sig")
    audits = {str(seed): audit_one(split, seed) for seed, split in seed_results.items()}
    audit = audits[str(primary_seed)] | {"split_file": str(OUT.relative_to(ROOT))}
    (ROOT / "results" / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (ROOT / "results" / "repeated_split_audit.json").write_text(
        json.dumps({"seeds": list(seeds), "split_file": str(REPEATED_OUT.relative_to(ROOT)), "audits": audits}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
