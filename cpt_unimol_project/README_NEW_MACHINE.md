# Legacy CPT Uni-Mol v1 Pipeline: Migration Guide

> This guide documents the earlier `phase2_unimol` pipeline, which loads Uni-Mol v1 (`model_name="unimolv1"`). It is not the Uni-Mol2 benchmark. For the current Uni-Mol2 84M evaluation and its results, start at [`activity_benchmark/README.md`](activity_benchmark/README.md).

The commands below reproduce the legacy setup documented for the 2026-03 snapshot; they do not reproduce the newer Uni-Mol2 Stage 7/12 benchmark.

## What To Keep

- Historical Uni-Mol v1 run target:
  - `cpt_unimol_project/phase2_unimol/artifacts_unimol_opt_cached_fast_b128_r3/`
- Earlier Uni-Mol v1 baseline (kept for reference):
  - `cpt_unimol_project/phase2_unimol/artifacts_unimol/`

Other intermediate debug/tuning outputs were intentionally cleaned.

## Quick Start (Windows)

1. Create and activate virtual environment:

```powershell
cd D:/kimi2.5program/JMC
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install pandas==1.5.3 numpy==1.26.4 scipy==1.13.1 scikit-learn==1.6.1 rdkit==2025.9.2 tqdm==4.67.3 requests==2.32.5 torch==2.8.0 unimol-tools==0.1.3
```

3. Build Phase1 table and conformer cache (run once):

```powershell
python cpt_unimol_project/phase1_3d/build_master_table.py
python cpt_unimol_project/phase1_3d/generate_conformers.py
```

4. Run the stable training command:

```powershell
python cpt_unimol_project/phase2_unimol/train_unimol_multitask.py --epochs 20 --batch_size 128 --early_stopping 4 --weight_decay 5e-5 --dropout 0.08 --freeze_strategy first_n --freeze_n_layers 8 --max_rows 0 --split scaffold --kfold 1 --num_workers 0 --max_atoms 120 --run_name artifacts_unimol_opt_cached_fast_b128_r3
```

## Why This Configuration

- `kfold=1`: largest speed gain on CPU.
- `batch_size=128`: reduces steps per epoch.
- `max_atoms=120`: avoids rare memory crash in `scipy.spatial.distance_matrix` on large molecules.
- Cache-only 3D path: when cache is incomplete, rows without cached conformers are dropped instead of triggering on-the-fly regeneration.

## Known Pitfalls (And Fixes)

- Training unexpectedly regenerates conformers:
  - Ensure `cpt_unimol_project/data/conformer_features.pkl` exists.
  - Current script uses cache-first logic and drops uncached rows.

- Epoch looks much slower after switching configs:
  - Check `kfold`: `kfold=1` changes step count and total data usage behavior.
  - Compare speed using per-step seconds, not only ETA.

- CPU memory error during preprocessing:
  - Use `--max_atoms 120` (or smaller if machine RAM is tight).

- No GPU acceleration:
  - The script auto-detects CUDA. If `torch.cuda.is_available()` is false, run is CPU-only.

## Minimal Validation Before Full Training

```powershell
python cpt_unimol_project/phase2_unimol/train_unimol_multitask.py --epochs 1 --batch_size 128 --max_rows 256 --split scaffold --kfold 1 --num_workers 0 --max_atoms 120 --run_name smoke_check
```

Expected signs:
- Message about cached 3D features loaded (or dropped uncached rows).
- `Kfold is 1` in logs.
- Training starts without conformer regeneration.

## Migration Checklist

- Keep these files:
  - `cpt_unimol_project/data/master_multitask.csv`
  - `cpt_unimol_project/data/conformer_features.pkl`
  - `cpt_unimol_project/phase2_unimol/train_unimol_multitask.py`
  - `cpt_unimol_project/phase2_unimol/predict_dual_activity.py`
- Recreate `.venv` and reinstall dependencies.
- Run smoke check first, then full training.

## For AI Agents

Use this command template directly and do not enable on-the-fly conformer generation for full runs:

```powershell
python cpt_unimol_project/phase2_unimol/train_unimol_multitask.py --epochs 20 --batch_size 128 --early_stopping 4 --weight_decay 5e-5 --dropout 0.08 --freeze_strategy first_n --freeze_n_layers 8 --max_rows 0 --split scaffold --kfold 1 --num_workers 0 --max_atoms 120 --run_name <new_run_name>
```
