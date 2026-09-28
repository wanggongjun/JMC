#!/usr/bin/env python3
"""Create one scaffold-to-fold map shared by both cancer-cell endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import assay_context_3d_benchmark as base

ROOT = base.ROOT


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-assay-molecules", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(base.SOURCE, encoding="utf-8-sig")
    raw = raw.loc[(raw.standard_type.astype(str).str.upper() == "IC50")
                  & (raw.standard_relation.astype(str) == "=")
                  & raw.pchembl_value_final.notna()].copy()
    raw["task"] = raw.cell_line.map(base.TASKS)
    raw = raw.dropna(subset=["task", "assay_chembl_id", "canonical_smiles"]).copy()
    raw["smiles"] = raw.canonical_smiles.astype(str)
    raw["y"] = pd.to_numeric(raw.pchembl_value_final, errors="coerce")
    raw = raw.dropna(subset=["y"])
    feature = np.load(base.FEAT2, allow_pickle=False)
    valid = set(feature["smiles"].astype(str))
    raw = raw.loc[raw.smiles.isin(valid)].copy()
    counts = raw.groupby(["task", "assay_chembl_id"]).smiles.nunique().rename("n").reset_index()
    kept = counts.loc[counts.n >= args.min_assay_molecules, ["task", "assay_chembl_id"]]
    raw = raw.merge(kept, on=["task", "assay_chembl_id"], how="inner", validate="many_to_one")
    if raw.empty:
        raise RuntimeError("No eligible assay records after the frozen filters")
    raw["scaffold"] = raw.smiles.map(scaffold)

    # Assign each scaffold once, balancing total assay records across both tasks.
    loads_by_scaffold = raw.groupby("scaffold").size().rename("record_count").reset_index()
    rng = np.random.default_rng(args.seed)
    loads_by_scaffold["tie"] = rng.random(len(loads_by_scaffold))
    loads_by_scaffold = loads_by_scaffold.sort_values(
        ["record_count", "tie"], ascending=[False, True]).reset_index(drop=True)
    loads = np.zeros(args.folds, dtype=np.int64)
    assignment: dict[str, int] = {}
    for row in loads_by_scaffold.itertuples(index=False):
        choices = np.flatnonzero(loads == loads.min())
        fold = int(rng.choice(choices))
        assignment[str(row.scaffold)] = fold
        loads[fold] += int(row.record_count)

    molecules = raw[["task", "smiles", "scaffold"]].drop_duplicates()
    molecules["outer_fold"] = molecules.scaffold.map(assignment).astype(int)
    if molecules.groupby("scaffold").outer_fold.nunique().max() != 1:
        raise RuntimeError("A scaffold received multiple fold assignments")
    if molecules.groupby("smiles").outer_fold.nunique().max() != 1:
        raise RuntimeError("A molecule received multiple fold assignments across tasks")
    molecules = molecules.sort_values(["task", "outer_fold", "smiles"]).reset_index(drop=True)
    molecules.to_csv(out, index=False)

    task_stats = {}
    for task, part in molecules.groupby("task", sort=True):
        task_records = raw.loc[raw.task == task]
        task_stats[str(task)] = {
            "molecules": int(part.smiles.nunique()),
            "scaffolds": int(part.scaffold.nunique()),
            "records_per_fold": [int((task_records.scaffold.map(assignment) == f).sum())
                                 for f in range(args.folds)],
            "scaffolds_per_fold": [int(part.loc[part.outer_fold == f, "scaffold"].nunique())
                                   for f in range(args.folds)],
        }
    audit = {
        "seed": int(args.seed), "folds": int(args.folds),
        "assignment_unit": "Bemis-Murcko scaffold; one global map shared by HepG2 and HCT116",
        "balancing": "greedy total source-record load across both tasks, randomized tie-breaking",
        "min_unique_molecules_per_task_assay": int(args.min_assay_molecules),
        "source_sha256": base.sha256(base.SOURCE), "features_2d_sha256": base.sha256(base.FEAT2),
        "manifest": str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),
        "manifest_sha256": sha256(out), "global_scaffolds": int(len(assignment)),
        "combined_record_load_per_fold": loads.astype(int).tolist(), "tasks": task_stats,
        "cross_task_scaffold_leakage": 0, "cross_task_molecule_fold_disagreement": 0,
    }
    out.with_suffix(out.suffix + ".audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
