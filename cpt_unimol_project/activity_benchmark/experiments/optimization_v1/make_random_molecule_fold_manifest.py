#!/usr/bin/env python3
"""Create molecule-grouped random folds for known-scaffold interpolation tests."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
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

    f2 = np.load(base.FEAT2, allow_pickle=False)
    f3_path = ROOT / "results/unimol2_plus_rdkit_rich_3d_features.npz"
    f3 = np.load(f3_path, allow_pickle=False)
    set2 = set(f2["smiles"].astype(str))
    set3 = set(f3["smiles"].astype(str))
    if set2 != set3:
        raise RuntimeError("2D/3D features do not have identical molecule coverage")
    raw = raw.loc[raw.smiles.isin(set2)].copy()
    assay_counts = raw.groupby(["task", "assay_chembl_id"]).smiles.nunique().rename("assay_n").reset_index()
    kept = assay_counts.loc[assay_counts.assay_n >= args.min_assay_molecules,
                            ["task", "assay_chembl_id"]]
    raw = raw.merge(kept, on=["task", "assay_chembl_id"], how="inner", validate="many_to_one")

    rows = []
    audit = {"seed": int(args.seed), "n_folds": int(args.n_folds), "split_unit": "canonical SMILES within each cell-line task",
             "scaffold_overlap_expected": True, "source_sha256": base.sha256(base.SOURCE),
             "assay_minimum_unique_molecules": int(args.min_assay_molecules), "tasks": {}}
    for task_i, (task, part) in enumerate(raw.groupby("task", sort=True)):
        molecule_counts = part.groupby("smiles").size().rename("records").reset_index()
        rng = np.random.default_rng(args.seed + 1009 * task_i)
        tie_order = rng.random(len(molecule_counts))
        molecule_counts["tie_order"] = tie_order
        molecule_counts = molecule_counts.sort_values(["records", "tie_order"], ascending=[False, True])
        loads = np.zeros(args.n_folds, dtype=np.int64)
        assignments = {}
        for row in molecule_counts.itertuples(index=False):
            minimum = loads.min()
            choices = np.flatnonzero(loads == minimum)
            fold = int(rng.choice(choices))
            assignments[str(row.smiles)] = fold
            loads[fold] += int(row.records)
        task_smiles = sorted(assignments)
        task_scaffolds = {smiles: scaffold(smiles) for smiles in task_smiles}
        for smiles in task_smiles:
            rows.append({"task": str(task), "smiles": smiles, "scaffold": task_scaffolds[smiles],
                         "outer_fold": int(assignments[smiles])})
        overlap_counts = []
        for fold in range(args.n_folds):
            train_scaffolds = {task_scaffolds[s] for s, f in assignments.items() if f != fold}
            test_scaffolds = {task_scaffolds[s] for s, f in assignments.items() if f == fold}
            overlap_counts.append(len(train_scaffolds & test_scaffolds))
        audit["tasks"][str(task)] = {"molecules": int(len(assignments)), "record_load_per_fold": loads.astype(int).tolist(),
                                      "scaffold_overlap_groups_per_fold": overlap_counts,
                                      "molecule_overlap_groups_per_fold": [0] * args.n_folds}

    manifest = pd.DataFrame(rows).sort_values(["task", "outer_fold", "smiles"]).reset_index(drop=True)
    manifest.to_csv(out, index=False)
    audit["manifest"] = str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)
    audit["manifest_sha256"] = base.sha256(out)
    out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
