#!/usr/bin/env python3
"""Join frozen Uni-Mol2 CLS and RDKit ensemble geometry features by canonical SMILES."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unimol", default="results/unimol2_84m_ensemble_mean_cls_repr.npz")
    parser.add_argument("--rdkit", default="results/rdkit_rich_3d_ensemble.npz")
    parser.add_argument("--embedding-view", choices=("mean", "boltzmann", "both"), default="mean")
    parser.add_argument("--output", default="results/unimol2_plus_rdkit_rich_3d_features.npz")
    args = parser.parse_args()

    paths = {}
    for name in ("unimol", "rdkit", "output"):
        path = Path(getattr(args, name))
        paths[name] = path if path.is_absolute() else ROOT / path
    unimol = np.load(paths["unimol"], allow_pickle=False)
    rdkit = np.load(paths["rdkit"], allow_pickle=False)
    u_smiles = unimol["smiles"].astype(str)
    r_smiles = rdkit["smiles"].astype(str)
    if len(set(u_smiles)) != len(u_smiles) or len(set(r_smiles)) != len(r_smiles):
        raise ValueError("Input feature files contain duplicate SMILES keys")
    if set(u_smiles) != set(r_smiles):
        raise ValueError("Uni-Mol2 and RDKit feature files do not have identical molecule coverage")

    r_index = {smiles: i for i, smiles in enumerate(r_smiles)}
    order = np.asarray([r_index[smiles] for smiles in u_smiles], dtype=np.int64)
    if args.embedding_view == "mean":
        cls = unimol["cls_repr"].astype(np.float32)
    elif args.embedding_view == "boltzmann":
        cls = unimol["boltzmann_cls_repr"].astype(np.float32)
    else:
        cls = np.concatenate([unimol["cls_repr"], unimol["boltzmann_cls_repr"]], axis=1).astype(np.float32)
    geometry = rdkit["features"][order].astype(np.float32)
    if not np.isfinite(cls).all() or not np.isfinite(geometry).all():
        raise ValueError("Input feature arrays contain non-finite values")
    combined = np.concatenate([cls, geometry], axis=1)
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["output"], smiles=u_smiles, features=combined)

    metadata = {
        "feature_name": "Uni-Mol2 conformer-pool CLS concatenated with RDKit 3D conformer-ensemble descriptors",
        "unimol_embedding_view": args.embedding_view,
        "unimol_aggregation": "arithmetic mean CLS" if args.embedding_view == "mean" else "298.15 K MMFF94s/UFF Boltzmann-weighted mean CLS" if args.embedding_view == "boltzmann" else "both arithmetic and 298.15 K MMFF94s/UFF Boltzmann-weighted mean CLS",
        "molecule_count": int(len(u_smiles)),
        "unimol2_dimension": int(cls.shape[1]),
        "rdkit_3d_dimension": int(geometry.shape[1]),
        "combined_dimension": int(combined.shape[1]),
        "join_key": "canonical SMILES string; exact key-set equality required",
        "input_files": {
            name: {"path": str(paths[name].relative_to(ROOT)), "sha256": sha256(paths[name])}
            for name in ("unimol", "rdkit")
        },
        "output_sha256": sha256(paths["output"]),
        "scaling": "no global data-dependent scaling; benchmark pipelines fit StandardScaler inside each training fold",
    }
    metadata_path = paths["output"].with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
