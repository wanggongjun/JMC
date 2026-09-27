"""Build an auditable conformer-ensemble descriptor set for activity modeling.

Features include RDKit shape, WHIM, USR/USRCAT and 3D autocorrelation
descriptors, plus heavy-atom solvent-accessible surface area partitions. Each
conformer is described separately, then pooled by unweighted mean, population
standard deviation and force-field Boltzmann mean. No labels enter this step.
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
from rdkit.Chem import AllChem, Descriptors3D, rdFreeSASA, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/modeling_dataset.csv"
ENSEMBLE = ROOT / "experiments/optimization_v1/conformer_ensemble_3x.pkl"
FALLBACK = ROOT / "data/processed/conformer_features.pkl"
KB_KCAL = 0.00198720425864083
TEMP_K = 298.15
PROBE_RADIUS = 1.4


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


def describe_conformer(mol: Chem.Mol, xyz: np.ndarray) -> tuple[list[float], list[str]]:
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, point in enumerate(xyz):
        conf.SetAtomPosition(i, tuple(float(v) for v in point))
    conf_id = mol.AddConformer(conf, assignId=True)

    values: list[float] = []
    names: list[str] = []
    blocks = [
        ("shape", [name for name, _ in Descriptors3D.descList],
         [fn(mol) for _, fn in Descriptors3D.descList]),
        ("whim", None, rdMolDescriptors.CalcWHIM(mol, conf_id)),
        ("usr", None, rdMolDescriptors.GetUSR(mol, conf_id)),
        ("usrcat", None, rdMolDescriptors.GetUSRCAT(mol, confId=conf_id)),
        ("autocorr3d", None, rdMolDescriptors.CalcAUTOCORR3D(mol, conf_id)),
    ]
    for block, explicit_names, block_values in blocks:
        if explicit_names is None:
            explicit_names = [f"{block}_{i:03d}" for i in range(len(block_values))]
        if len(explicit_names) != len(block_values):
            raise RuntimeError(f"Descriptor-name length mismatch in {block}")
        names.extend(f"{block}_{name}" for name in explicit_names)
        values.extend(float(x) for x in block_values)

    pt = Chem.GetPeriodicTable()
    radii = [float(pt.GetRvdw(atom.GetAtomicNum()) + PROBE_RADIUS)
             for atom in mol.GetAtoms()]
    rdFreeSASA.CalcSASA(mol, radii, conf_id)
    heavy_areas = []
    polar_areas = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        area = float(atom.GetProp("SASA"))
        heavy_areas.append(area)
        if atom.GetAtomicNum() in (7, 8, 15, 16):
            polar_areas.append(area)
    total = float(np.sum(heavy_areas))
    polar = float(np.sum(polar_areas))
    sasa_values = [total, polar, total - polar, polar / max(total, 1e-12)]
    sasa_names = ["heavy_total", "heavy_polar", "heavy_apolar", "heavy_polar_fraction"]
    names.extend(f"sasa_{name}" for name in sasa_names)
    values.extend(sasa_values)

    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite 3D descriptor")
    return values, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test rows only")
    parser.add_argument("--output", default="results/rdkit_rich_3d_ensemble.npz")
    args = parser.parse_args()
    started = time.time()
    data = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    if args.limit:
        data = data.head(args.limit).copy()
    with ENSEMBLE.open("rb") as f:
        entries = pickle.load(f)
    by_smiles = {str(x["smiles"]): x for x in entries}
    with FALLBACK.open("rb") as f:
        fallback_rows = pickle.load(f)
    fallback_by_smiles = {str(x["smiles"]): x for x in fallback_rows}

    matrices: list[np.ndarray] = []
    counts: list[int] = []
    quality: list[str] = []
    columns: list[str] | None = None
    for row_i, smiles in enumerate(data.smiles.astype(str)):
        entry = by_smiles.get(smiles)
        confs = [] if entry is None else entry.get("conformers", [])
        expected_atoms = [] if entry is None else entry.get("atoms", [])
        row_quality = str(entry.get("quality_status", "unknown")) if entry else "unknown"
        if not confs:
            prior = fallback_by_smiles.get(smiles)
            if prior is None or prior.get("atoms") is None or prior.get("coords") is None:
                raise RuntimeError(f"No conformer found at row {row_i}: {smiles}")
            confs = [{"coordinates": prior["coords"], "energy": float("nan")}]
            expected_atoms = prior["atoms"]
            row_quality = "original_cache_fallback"
        mol = largest_hydrogenated_mol(smiles)
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if atom_symbols != expected_atoms:
            raise RuntimeError(f"Atom-order mismatch at row {row_i}: {smiles}")

        per_conf = []
        energies = []
        for item in confs:
            xyz = np.asarray(item["coordinates"], dtype=np.float64)
            if xyz.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(xyz).all():
                raise RuntimeError(f"Invalid geometry at row {row_i}: {xyz.shape}")
            values, names = describe_conformer(mol, xyz)
            if columns is None:
                columns = names
            elif columns != names:
                raise RuntimeError("Inconsistent descriptor ordering")
            per_conf.append(values)
            energies.append(float(item.get("energy", np.nan)))

        array = np.asarray(per_conf, dtype=np.float64)
        energy_array = np.asarray(energies, dtype=np.float64)
        means = array.mean(axis=0)
        stds = array.std(axis=0, ddof=0)
        valid_energy = np.isfinite(energy_array)
        if valid_energy.any():
            relative = energy_array[valid_energy] - np.min(energy_array[valid_energy])
            weights = np.exp(-relative / (KB_KCAL * TEMP_K))
            weights /= weights.sum()
            boltz = np.average(array[valid_energy], axis=0, weights=weights)
            conformer_entropy = -float(np.sum(weights * np.log(np.clip(weights, 1e-15, 1.0))))
            energy_span = float(np.max(energy_array[valid_energy]) - np.min(energy_array[valid_energy]))
            energy_available = 1.0
        else:
            boltz = means.copy()
            conformer_entropy = 0.0
            energy_span = 0.0
            energy_available = 0.0
        pooled = np.concatenate([means, stds, boltz,
                                 [conformer_entropy, energy_span, energy_available]])
        matrices.append(pooled.astype(np.float32))
        counts.append(len(per_conf))
        quality.append(row_quality)
        if (row_i + 1) % 500 == 0 or row_i + 1 == len(data):
            print(f"described {row_i + 1}/{len(data)}", flush=True)

    feature_matrix = np.asarray(matrices, dtype=np.float32)
    if columns is None or feature_matrix.shape != (len(data), 3 * len(columns) + 3):
        raise RuntimeError(f"Unexpected feature shape: {feature_matrix.shape}")
    if not np.isfinite(feature_matrix).all():
        raise RuntimeError("Non-finite pooled feature matrix")
    pooled_columns = ([f"mean_{x}" for x in columns]
                      + [f"std_{x}" for x in columns]
                      + [f"boltz_{x}" for x in columns]
                      + ["conformer_entropy", "forcefield_energy_span", "forcefield_energy_available"])
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, smiles=np.asarray(data.smiles.astype(str).tolist(), dtype=str),
                        features=feature_matrix, columns=np.asarray(pooled_columns),
                        conformer_counts=np.asarray(counts, dtype=np.int16))
    metadata = {
        "descriptor_source": "RDKit Descriptors3D, WHIM, USR, USRCAT, AUTOCORR3D and FreeSASA",
        "descriptor_base_count": len(columns), "feature_shape": list(feature_matrix.shape),
        "aggregation": "conformer mean, population std and force-field Boltzmann mean; plus conformer entropy and force-field energy span",
        "boltzmann_temperature_kelvin": TEMP_K,
        "sasa_radii": "RDKit van der Waals radius + 1.4 A solvent probe; heavy-atom total/polar/apolar exposed surface",
        "input_molecules": len(data), "conformer_count_min": int(min(counts)),
        "conformer_count_median": float(np.median(counts)), "conformer_count_max": int(max(counts)),
        "quality_status_counts": pd.Series(quality).value_counts().to_dict(),
        "modeling_dataset_sha256": sha256(DATA), "conformer_ensemble_sha256": sha256(ENSEMBLE),
        "fallback_conformer_cache_sha256": sha256(FALLBACK), "rdkit_version": rdBase.rdkitVersion,
        "smoke_test_limit": args.limit or None, "elapsed_seconds": round(time.time() - started, 2),
    }
    out.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
