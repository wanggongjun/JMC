"""Create conformer-ensemble RDKit 3D shape descriptors for a matched benchmark.

For each molecule, calculate the 11 RDKit Descriptors3D on every retained
ETKDG/MMFF94s-or-UFF conformer and save conformer mean, standard deviation,
and force-field Boltzmann-weighted mean. This is a low-cost 3D baseline and
can be concatenated with the existing 2D features in a nested scaffold CV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors3D

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/modeling_dataset.csv"
ENSEMBLE = ROOT / "experiments/optimization_v1/conformer_ensemble_3x.pkl"
FALLBACK = ROOT / "data/processed/conformer_features.pkl"
KB_KCAL = 0.00198720425864083
TEMP_K = 298.15


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def largest_hydrogenated_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return Chem.AddHs(mol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test rows only")
    parser.add_argument("--conformer-file", default=str(ENSEMBLE),
                        help="Conformer pickle; defaults to the established 3-conformer ensemble")
    parser.add_argument("--summary", choices=("mean_std_boltzmann", "mean_boltzmann"),
                        default="mean_std_boltzmann",
                        help="Use all standard summaries or the compact mean + Boltzmann view")
    parser.add_argument("--output", default="results/rdkit_3d_shape_ensemble.npz")
    args = parser.parse_args()
    started = time.time()
    data = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    if args.limit:
        data = data.head(args.limit).copy()
    ensemble_path = Path(args.conformer_file)
    if not ensemble_path.is_absolute():
        ensemble_path = ROOT / ensemble_path
    with ensemble_path.open("rb") as f:
        entries = pickle.load(f)
    by_smiles = {str(x["smiles"]): x for x in entries}
    with FALLBACK.open("rb") as f:
        fallback_rows = pickle.load(f)
    fallback_by_smiles = {str(x["smiles"]): x for x in fallback_rows}
    names = [name for name, _ in Descriptors3D.descList]
    functions = [fn for _, fn in Descriptors3D.descList]
    all_features = []
    conformer_counts = []
    quality = []
    for row_i, smi in enumerate(data.smiles.astype(str)):
        entry = by_smiles.get(smi)
        confs = [] if entry is None else entry.get("conformers", [])
        expected_atoms = [] if entry is None else entry.get("atoms", [])
        row_quality = "ensemble"
        if not confs:
            prior = fallback_by_smiles.get(smi)
            if prior is None or prior.get("atoms") is None or prior.get("coords") is None:
                raise RuntimeError(f"Missing ensemble and original fallback at row {row_i}: {smi}")
            confs = [{"coordinates": prior["coords"], "energy": np.nan}]
            expected_atoms = prior["atoms"]
            row_quality = "original_cache_fallback"
        mol = largest_hydrogenated_mol(smi)
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if atom_symbols != expected_atoms:
            raise RuntimeError(f"Atom-order mismatch at row {row_i}: {smi}")
        vals, energies = [], []
        for item in confs:
            xyz = np.asarray(item["coordinates"], dtype=np.float64)
            if xyz.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(xyz).all():
                raise RuntimeError(f"Invalid conformer geometry at row {row_i}: {xyz.shape}")
            conf = Chem.Conformer(mol.GetNumAtoms())
            for atom_i, point in enumerate(xyz):
                conf.SetAtomPosition(atom_i, tuple(float(v) for v in point))
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            row = [float(fn(mol)) for fn in functions]
            if not np.isfinite(row).all():
                raise RuntimeError(f"Non-finite 3D descriptor at row {row_i}: {smi}")
            vals.append(row)
            energies.append(float(item.get("energy", np.nan)))
        a = np.asarray(vals, dtype=np.float64)
        mean = a.mean(axis=0)
        std = a.std(axis=0, ddof=0)
        e = np.asarray(energies, dtype=np.float64)
        valid_e = np.isfinite(e)
        if valid_e.any():
            rel = e[valid_e] - e[valid_e].min()
            weights = np.exp(-rel / (KB_KCAL * TEMP_K))
            weights /= weights.sum()
            boltz = np.average(a[valid_e], axis=0, weights=weights)
        else:
            boltz = mean.copy()
        summaries = [mean, std, boltz] if args.summary == "mean_std_boltzmann" else [mean, boltz]
        all_features.append(np.concatenate(summaries).astype(np.float32))
        conformer_counts.append(len(vals))
        quality.append(row_quality if row_quality != "ensemble" else str(entry.get("quality_status", "unknown")))
        if (row_i + 1) % 500 == 0 or row_i + 1 == len(data):
            print(f"described {row_i + 1}/{len(data)}", flush=True)

    matrix = np.asarray(all_features, dtype=np.float32)
    multiplier = 3 if args.summary == "mean_std_boltzmann" else 2
    if matrix.shape != (len(data), len(names) * multiplier) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Unexpected descriptor matrix: {matrix.shape}")
    columns = [f"mean_{n}" for n in names]
    if args.summary == "mean_std_boltzmann":
        columns += [f"std_{n}" for n in names]
    columns += [f"boltzmann_{n}" for n in names]
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, smiles=np.asarray(data.smiles.astype(str).tolist(), dtype=str), features=matrix,
                        columns=np.asarray(columns), conformer_counts=np.asarray(conformer_counts, dtype=np.int16))
    metadata = {
        "descriptor_source": "RDKit Descriptors3D",
        "descriptor_names": names,
        "aggregation": ("mean, population std, and 298.15 K force-field Boltzmann-weighted mean across retained conformers"
                        if args.summary == "mean_std_boltzmann" else
                        "mean and 298.15 K force-field Boltzmann-weighted mean across retained conformers"),
        "boltzmann_energy_source": "per-molecule MMFF94s/UFF conformer energies from the fixed ETKDGv3 ensemble",
        "temperature_kelvin": TEMP_K,
        "input_molecules": len(data),
        "feature_shape": list(matrix.shape),
        "conformer_count_min": int(min(conformer_counts)),
        "conformer_count_median": float(np.median(conformer_counts)),
        "conformer_count_max": int(max(conformer_counts)),
        "quality_status_counts": pd.Series(quality).value_counts().to_dict(),
        "modeling_dataset_sha256": sha256(DATA),
        "conformer_ensemble_file": str(ensemble_path.relative_to(ROOT)) if ensemble_path.is_relative_to(ROOT) else str(ensemble_path),
        "conformer_ensemble_sha256": sha256(ensemble_path),
        "fallback_conformer_cache_sha256": sha256(FALLBACK),
        "rdkit_version": rdBase.rdkitVersion,
        "smoke_test_limit": args.limit or None,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    out.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
