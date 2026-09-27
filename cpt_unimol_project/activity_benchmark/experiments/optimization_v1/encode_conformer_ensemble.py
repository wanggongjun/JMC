"""Mean-pool frozen Uni-Mol CLS vectors over a generated conformer ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("UNIMOL_WEIGHT_DIR", str(ROOT / "models/unimol"))
from unimol_tools import UniMolRepr

DATA = ROOT / "data/processed/modeling_dataset.csv"
ENSEMBLE = ROOT / "experiments/optimization_v1/conformer_ensemble_3x.pkl"
FALLBACK = ROOT / "data/processed/conformer_features.pkl"
WEIGHTS = ROOT / "models/unimol/mol_pre_all_h_220816.pt"
DICT = ROOT / "models/unimol/mol.dict.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformer-file", default=str(ENSEMBLE))
    parser.add_argument("--output", default="results/unimol_v1_ensemble_mean_cls_repr.npz")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test row limit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    started = time.time()
    data = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    if args.limit:
        data = data.head(args.limit).copy()
    conf_path = Path(args.conformer_file)
    if not conf_path.is_absolute():
        conf_path = ROOT / conf_path
    with conf_path.open("rb") as f:
        conformers = pickle.load(f)
    with FALLBACK.open("rb") as f:
        fallback_rows = pickle.load(f)
    conf_by_smiles = {str(row["smiles"]): row for row in conformers}
    fallback_by_smiles = {str(row["smiles"]): row for row in fallback_rows}
    smiles = data.smiles.astype(str).tolist()
    flat_smiles, flat_atoms, flat_coords, row_positions = [], [], [], []
    fallback_counts = np.zeros(len(smiles), dtype=np.int16)
    n_conformers = np.zeros(len(smiles), dtype=np.int16)
    for row_i, smi in enumerate(smiles):
        entry = conf_by_smiles.get(smi)
        usable = [] if entry is None else entry.get("conformers", [])
        atoms = [] if entry is None else entry.get("atoms", [])
        if not usable:
            prior = fallback_by_smiles.get(smi)
            if prior and prior.get("atoms") is not None and prior.get("coords") is not None:
                usable = [{"coordinates": np.asarray(prior["coords"], dtype=np.float32)}]
                atoms = prior["atoms"]
                fallback_counts[row_i] = 1
        if not usable or not atoms:
            raise RuntimeError(f"No usable conformer for {smi}; refusing to silently drop a matched row")
        n_conformers[row_i] = len(usable)
        for item in usable:
            xyz = np.asarray(item["coordinates"], dtype=np.float32)
            if xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all():
                raise RuntimeError(f"Invalid conformer geometry for {smi}: {xyz.shape}")
            flat_smiles.append(smi)
            flat_atoms.append(atoms)
            flat_coords.append(xyz)
            row_positions.append(row_i)

    torch.set_num_threads(max(1, min(4, int(os.environ.get("OMP_NUM_THREADS", "4")))))
    chosen_device = args.device
    if chosen_device == "auto":
        chosen_device = "mps" if torch.backends.mps.is_available() else "cpu"
    repr_model = UniMolRepr(
        data_type="molecule", batch_size=args.batch_size, remove_hs=False,
        model_name="unimolv1", use_cuda=False, max_atoms=320,
    )
    if chosen_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        from unimol_tools.tasks import Trainer
        original_init = Trainer.__init__

        def mps_trainer_init(self, *a, **kw):
            original_init(self, *a, **kw)
            self.device = torch.device("mps")

        def mps_decorate(self, batch):
            net_input, net_target = batch
            if isinstance(net_input, dict):
                net_input = {k: v.to(self.device) for k, v in net_input.items()}
            else:
                net_input = {"net_input": net_input.to(self.device)}
            if self.task == "repr":
                net_target = None
            else:
                net_target = net_target.to(self.device).float()
            return net_input, net_target

        Trainer.__init__ = mps_trainer_init
        Trainer.decorate_torch_batch = mps_decorate
        repr_model.device = torch.device("mps")
        repr_model.model = repr_model.model.to(repr_model.device)
    custom_data = {"SMILES": flat_smiles, "atoms": flat_atoms, "coordinates": flat_coords}
    reprs = np.asarray(repr_model.get_repr(custom_data, return_atomic_reprs=False), dtype=np.float32)
    if reprs.ndim != 2 or reprs.shape[0] != len(flat_smiles):
        raise RuntimeError(f"Unexpected Uni-Mol representation shape: {reprs.shape}")
    summed = np.zeros((len(smiles), reprs.shape[1]), dtype=np.float64)
    np.add.at(summed, np.asarray(row_positions, dtype=np.int64), reprs.astype(np.float64))
    mean_repr = (summed / n_conformers[:, None]).astype(np.float32)
    if not np.isfinite(mean_repr).all():
        raise RuntimeError("Mean-pooled Uni-Mol representations contain non-finite values")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, smiles=np.asarray(smiles), cls_repr=mean_repr,
                        n_conformers=n_conformers, fallback_single=fallback_counts)
    metadata = {
        "model_name": "unimolv1",
        "unimol_tools_version": version("unimol-tools"),
        "pretrained": True,
        "fine_tuned": False,
        "aggregation": "arithmetic mean of per-conformer molecule CLS embeddings",
        "input_molecules": len(smiles),
        "input_conformers": len(flat_smiles),
        "embedding_shape": list(mean_repr.shape),
        "conformer_count_min": int(n_conformers.min()),
        "conformer_count_median": float(np.median(n_conformers)),
        "conformer_count_max": int(n_conformers.max()),
        "fallback_single_conformer_molecules": int(fallback_counts.sum()),
        "device": chosen_device,
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "conformer_file": str(conf_path.relative_to(ROOT)) if conf_path.is_relative_to(ROOT) else str(conf_path),
        "conformer_file_sha256": sha256(conf_path),
        "modeling_dataset_sha256": sha256(DATA),
        "pretrained_weight_sha256": sha256(WEIGHTS),
        "dictionary_sha256": sha256(DICT),
        "smoke_test_limit": args.limit or None,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
