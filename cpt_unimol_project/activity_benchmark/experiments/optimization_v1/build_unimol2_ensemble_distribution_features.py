#!/usr/bin/env python3
"""Build label-free Uni-Mol2 conformer mean and compact distribution summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/unimol2_84m_10conf_pooled_cls.npz"
DEFAULT_OUT = ROOT / "results/unimol2_ensemble_distribution_features.npz"
R_KCAL = 0.00198720425864083


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--temperature-k", type=float, default=298.15)
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = np.load(source, allow_pickle=False)
    smiles = archive["smiles"].astype(str)
    per_conf = archive["per_conf_repr"].astype(np.float32)
    mask = archive["conformer_mask"].astype(bool)
    energy = archive["conformer_energies"].astype(np.float64)
    mean_cls = archive["cls_repr"].astype(np.float32)
    boltzmann_cls = archive["boltzmann_cls_repr"].astype(np.float32)
    if per_conf.ndim != 3 or per_conf.shape[:2] != mask.shape or energy.shape != mask.shape:
        raise RuntimeError("Unexpected conformer array alignment")
    if not np.isfinite(per_conf[mask]).all() or not np.isfinite(mean_cls).all() or not np.isfinite(boltzmann_cls).all():
        raise RuntimeError("Non-finite Uni-Mol2 embeddings on retained conformers")
    stats = np.zeros((len(smiles), 6), dtype=np.float32)
    boltzmann_cls_check = np.zeros_like(boltzmann_cls)
    fallback_uniform = 0
    for i in range(len(smiles)):
        valid = np.flatnonzero(mask[i])
        if not len(valid):
            raise RuntimeError(f"No retained conformer for molecule index {i}")
        conf = per_conf[i, valid].astype(np.float64)
        centered = conf - conf.mean(axis=0, keepdims=True)
        if len(valid) > 1:
            gram = (centered @ centered.T) / float(len(valid) - 1)
            eig = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
            total = float(eig.sum())
            if total > 1e-12:
                p = eig[eig > total * 1e-12] / total
                entropy = float(-np.sum(p * np.log(p)))
                top = float(eig[-1])
                fro = float(np.sqrt(np.square(eig).sum()))
                effective_rank = float(np.exp(entropy))
                normalized_entropy = float(entropy / np.log(max(2, len(valid)-1)))
            else:
                top = fro = effective_rank = normalized_entropy = 0.0
        else:
            total = top = fro = effective_rank = normalized_entropy = 0.0

        valid_e = energy[i, valid]
        finite_e = np.isfinite(valid_e)
        if finite_e.any():
            selected_e = valid_e[finite_e]
            delta = selected_e - selected_e.min()
            weights = np.exp(-delta / (R_KCAL * args.temperature_k))
            weights /= weights.sum()
            valid_conf = conf[finite_e]
            pooled = np.sum(valid_conf * weights[:, None], axis=0)
            ess = float(1.0 / np.sum(weights ** 2))
            # Missing energy for a retained conformer is not silently treated as
            # a high-energy conformer. Such rare rows use an explicit uniform pool.
            if not finite_e.all():
                weights = np.ones(len(valid), dtype=np.float64) / len(valid)
                pooled = np.sum(conf * weights[:, None], axis=0)
                ess = float(len(valid))
                fallback_uniform += 1
        else:
            weights = np.ones(len(valid), dtype=np.float64) / len(valid)
            pooled = np.sum(conf * weights[:, None], axis=0)
            ess = float(len(valid))
            fallback_uniform += 1

        boltzmann_cls_check[i] = pooled.astype(np.float32)
        stats[i] = np.asarray([np.log1p(total), np.log1p(fro),
                               top / total if total > 1e-12 else 0.0,
                               effective_rank, normalized_entropy, ess], dtype=np.float32)

    if not np.allclose(boltzmann_cls_check, boltzmann_cls, rtol=2e-3, atol=2e-3):
        max_diff = float(np.max(np.abs(boltzmann_cls_check - boltzmann_cls)))
        raise RuntimeError(f"Recomputed Boltzmann pooling disagrees with archived vector: {max_diff}")
    features = np.concatenate([mean_cls, boltzmann_cls, stats], axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite distribution feature matrix")
    np.savez_compressed(out, smiles=smiles, features=features,
                        mean_cls=mean_cls, boltzmann_cls=boltzmann_cls,
                        distribution_stats=stats)
    metadata = {
        "method": "Uni-Mol2 84M frozen CLS ensemble mean + MMFF/UFF Boltzmann CLS + six compact ensemble-dispersion statistics",
        "source_npz": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "source_sha256": sha256(source),
        "output_npz": str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),
        "output_sha256": sha256(out),
        "n_molecules": int(len(smiles)), "n_conformers_present": int(mask.sum()),
        "feature_shape": list(features.shape), "temperature_kelvin": float(args.temperature_k),
        "stat_columns": ["log1p_covariance_trace", "log1p_covariance_frobenius_norm",
                         "top_eigenvalue_fraction", "effective_rank", "normalized_spectral_entropy",
                         "boltzmann_effective_conformer_count"],
        "uniform_pool_fallback_molecules": int(fallback_uniform),
        "labels_used": False,
    }
    out.with_suffix(out.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
