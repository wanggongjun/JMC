from pathlib import Path
from importlib.metadata import version
import argparse
import hashlib
import json
import os
import pickle
import time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import pandas as pd
import torch
from unimol_tools import UniMolRepr

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("UNIMOL_WEIGHT_DIR", str(ROOT / "models" / "unimol"))
DATA = ROOT / "data" / "processed" / "modeling_dataset.csv"
CACHE = ROOT / "data" / "processed" / "conformer_features.pkl"
OUT = ROOT / "results" / "unimol_v1_cls_repr.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test row limit; 0 means all 3D-usable molecules")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    df = pd.read_csv(DATA, encoding="utf-8-sig").sort_values("smiles").reset_index(drop=True)
    if args.limit > 0:
        df = df.head(args.limit).copy()
    with CACHE.open("rb") as f:
        entries = pickle.load(f)
    by_smiles = {str(item["smiles"]): item for item in entries}
    smiles = df.smiles.astype(str).tolist()
    selected = [by_smiles[s] for s in smiles]
    atoms = [item["atoms"] for item in selected]
    coordinates = [np.asarray(item["coords"], dtype=np.float32) for item in selected]
    max_atom_count = max(len(a) for a in atoms)
    if max_atom_count > 320:
        raise RuntimeError(f"Cached conformer exceeds the declared max_atoms=320: {max_atom_count}")
    repr_model = UniMolRepr(
        data_type="molecule", batch_size=8, remove_hs=False,
        model_name="unimolv1", use_cuda=False, max_atoms=320,
    )
    chosen_device = args.device
    if chosen_device == "auto":
        chosen_device = "mps" if torch.backends.mps.is_available() else "cpu"
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
    custom_data = {"SMILES": smiles, "atoms": atoms, "coordinates": coordinates}
    output = repr_model.get_repr(custom_data, return_atomic_reprs=False)
    cls = np.asarray(output, dtype=np.float32)
    if cls.ndim != 2 or cls.shape[0] != len(smiles):
        raise RuntimeError(f"Unexpected representation shape {cls.shape} for {len(smiles)} molecules")
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, smiles=np.asarray(smiles), cls_repr=cls)
    weight_dir = Path(os.environ["UNIMOL_WEIGHT_DIR"])
    metadata = {
        "model_name": "unimolv1",
        "unimol_tools_version": version("unimol-tools"),
        "pretrained": True,
        "fine_tuned": False,
        "representation": "molecule-level CLS embedding",
        "input_molecules": len(smiles),
        "smoke_test_limit": args.limit or None,
        "output_file": str(output_path.relative_to(ROOT)) if output_path.is_relative_to(ROOT) else str(output_path),
        "embedding_shape": list(cls.shape),
        "device": chosen_device,
        "cached_conformer_source": "RDKit ETKDG coordinates from data/processed/conformer_features.pkl",
        "max_cached_atom_count": max_atom_count,
        "max_atoms": 320,
        "batch_size": 8,
        "torch_version": torch.__version__,
        "modeling_dataset_sha256": sha256(DATA),
        "conformer_cache_sha256": sha256(CACHE),
        "conformer_status_counts": {str(k): int(v) for k, v in df.conformer_status.value_counts().items()},
        "pretrained_weight_file": "mol_pre_all_h_220816.pt",
        "pretrained_weight_sha256": sha256(weight_dir / "mol_pre_all_h_220816.pt"),
        "dictionary_sha256": sha256(weight_dir / "mol.dict.txt"),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
