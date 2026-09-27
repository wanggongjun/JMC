"""Generate deterministic, force-field-minimized conformer ensembles.

Outputs an experiment-specific pickle; the original single-conformer cache is
never overwritten. See --help for a small smoke-test mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/modeling_dataset.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def generate(smiles: str, row_id: int, count: int, base_seed: int) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"status": "invalid_smiles", "conformers": [], "atoms": [],
                "n_embedded": 0, "n_converged": 0}

    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    if mol.GetNumHeavyAtoms() > 70:
        return {"status": "skip_too_large", "conformers": [], "atoms": [],
                "n_embedded": 0, "n_converged": 0}

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int((base_seed + row_id) % (2**31 - 1))
    params.maxIterations = 1000
    params.numThreads = 1
    params.pruneRmsThresh = 0.35
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=count, params=params))
    if not conf_ids:
        return {"status": "embed_failed", "conformers": [], "atoms": [],
                "n_embedded": 0, "n_converged": 0}

    mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    ff_name = "MMFF94s" if mmff_props is not None else "UFF"
    optimized = []
    for conf_id in conf_ids:
        try:
            if mmff_props is not None:
                status = AllChem.MMFFOptimizeMolecule(
                    mol, mmffVariant="MMFF94s", confId=int(conf_id), maxIters=1000
                )
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol, mmff_props, confId=int(conf_id)
                )
            else:
                status = AllChem.UFFOptimizeMolecule(
                    mol, confId=int(conf_id), maxIters=1000
                )
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
            energy = float(ff.CalcEnergy()) if ff is not None else float("nan")
        except Exception:
            status, energy = -1, float("nan")
        conf = mol.GetConformer(int(conf_id))
        coords = np.asarray(conf.GetPositions(), dtype=np.float32)
        if coords.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(coords).all():
            continue
        optimized.append({
            "coordinates": coords,
            "energy": energy,
            "converged": bool(status == 0),
            "force_field": ff_name,
        })

    # Stable energy ordering, with valid but non-finite force-field cases last.
    optimized.sort(key=lambda item: (not np.isfinite(item["energy"]),
                                     item["energy"] if np.isfinite(item["energy"]) else float("inf")))
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    converged = sum(bool(x["converged"]) for x in optimized)
    quality = "all_converged" if optimized and converged == len(optimized) else "some_nonconverged"
    return {"status": "ok" if optimized else "no_valid_coordinates",
            "quality_status": quality, "conformers": optimized, "atoms": atoms,
            "force_field": ff_name, "n_embedded": len(conf_ids),
            "n_converged": converged}


def generate_job(job: tuple[str, int, int, int]) -> dict:
    return generate(*job)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-conformers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test row limit; 0 means all rows")
    parser.add_argument(
        "--output",
        default="experiments/optimization_v1/conformer_ensemble_3x.pkl",
    )
    args = parser.parse_args()
    if args.num_conformers < 1:
        raise ValueError("--num-conformers must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    started = time.time()
    data = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    if args.limit:
        data = data.head(args.limit).copy()
    entries = []
    counts = Counter()
    quality_counts = Counter()
    jobs = ((str(row.smiles), i, args.num_conformers, args.seed)
            for i, row in enumerate(data.itertuples(index=False)))
    if args.workers == 1:
        generated = map(generate_job, jobs)
        for i, (row, result) in enumerate(zip(data.itertuples(index=False), generated)):
            counts[result["status"]] += 1
            quality_counts[result.get("quality_status", "not_applicable")] += 1
            entries.append({"row_id": i, "smiles": str(row.smiles), **result})
            if (i + 1) % 100 == 0 or i + 1 == len(data):
                print(f"generated {i + 1}/{len(data)}; status_counts={dict(counts)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            generated = executor.map(generate_job, jobs, chunksize=1)
            for i, (row, result) in enumerate(zip(data.itertuples(index=False), generated)):
                counts[result["status"]] += 1
                quality_counts[result.get("quality_status", "not_applicable")] += 1
                entries.append({"row_id": i, "smiles": str(row.smiles), **result})
                if (i + 1) % 100 == 0 or i + 1 == len(data):
                    print(f"generated {i + 1}/{len(data)}; status_counts={dict(counts)}", flush=True)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "data_file": str(DATA.relative_to(ROOT)),
        "data_sha256": sha256(DATA),
        "output_file": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
        "input_rows": len(data),
        "num_conformers_requested": args.num_conformers,
        "seed": args.seed,
        "workers": args.workers,
        "embedding": "RDKit ETKDGv3; RMS pruning 0.35 A; hydrogens included",
        "optimization": "MMFF94s when parameters exist, otherwise UFF; 1000 iterations per conformer",
        "status_counts": dict(counts),
        "quality_status_counts": dict(quality_counts),
        "mean_conformers_generated": round(float(np.mean([x["n_embedded"] for x in entries])), 3),
        "mean_conformers_kept": round(float(np.mean([len(x["conformers"]) for x in entries])), 3),
        "rdkit_version": rdBase.rdkitVersion,
        "smoke_test_limit": args.limit or None,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
