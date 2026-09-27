# Uni-Mol2 Cell Activity Benchmark

This folder contains the code and validation record for an assay-aware comparison of 2D and Uni-Mol2 3D representations on ChEMBL HepG2 and HCT116 cell-based IC50 data. It follows the earlier `cpt_unimol_project` work and keeps the later scaffold-held-out benchmark separate.

## Result at a glance

One prespecified Stage 7 scaffold split (seed `20260927`) produced lower RMSE for the validation-selected 3D-bearing model family on both endpoints: `-0.00825` for HepG2 and `-0.00569` for HCT116 relative to the paired 2D model. These are preliminary point estimates. The other prespecified Stage 7 split (seed `20260926`) was worse on both endpoints, and both seed-20260927 scaffold-bootstrap intervals cross zero.

Stage 12 then evaluated conformer-distribution summaries with two fresh shared-scaffold manifests and both a matched nested 2D baseline and a stronger independent 2D reference. The prespecified gate did not pass: the 3D-bearing candidate did not beat both references on both cell lines and both seeds, and all paired bootstrap intervals cross zero. The current evidence does not establish a reproducible 3D advantage.

See [`reports/results_summary.md`](reports/results_summary.md) for the complete aggregate results, and [`experiments/optimization_v1/STAGE7_PROTOCOL.md`](experiments/optimization_v1/STAGE7_PROTOCOL.md) / [`STAGE12_PROTOCOL.md`](experiments/optimization_v1/STAGE12_PROTOCOL.md) for frozen methods and evaluation rules.

## Stage 7 exploratory result

RMSE is lower-is-better. Each seed is a five-fold scaffold manifest. Metrics are pooled known-assay outer-fold predictions; intervals use paired scaffold bootstrap.

| Seed | Cell line | Paired 2D RMSE | Selected 3D-bearing RMSE | Δ RMSE (95% scaffold-bootstrap CI) | Paired records |
|---|---|---:|---:|---:|---:|
| 20260926 | HepG2 | 0.63885 | 0.64754 | +0.00869 `[-0.00430,+0.02051]` | 1,516 |
| 20260926 | HCT116 | 0.73257 | 0.75024 | +0.01767 `[+0.00238,+0.03164]` | 1,248 |
| 20260927 | HepG2 | 0.66579 | 0.65754 | -0.00825 `[-0.03381,+0.01241]` | 1,478 |
| 20260927 | HCT116 | 0.71894 | 0.71326 | -0.00569 `[-0.01866,+0.00833]` | 1,221 |

The seed-20260927 direction did not repeat. It is accurate to describe that single run as an exploratory improvement; it is not accurate to generalize it as a confirmed dual-cell-line gain.

## Stage 12 confirmatory comparison

The candidate is the preselected 2D+Uni-Mol2 conformer-distribution late-fusion strategy. `Δ = candidate RMSE − baseline RMSE`.

| Seed | Cell line | Candidate | Matched 2D | Δ vs matched 2D (95% CI) | Independent 2D | Δ vs independent 2D (95% CI) |
|---|---|---:|---:|---:|---:|---:|
| 20261009 | HepG2 | 0.64094 | 0.64624 | -0.00530 `[-0.01408,+0.00294]` | 0.64981 | -0.00888 `[-0.02591,+0.00824]` |
| 20261009 | HCT116 | 0.68079 | 0.68119 | -0.00040 `[-0.00573,+0.00425]` | 0.69376 | -0.01297 `[-0.03059,+0.00446]` |
| 20261010 | HepG2 | 0.70651 | 0.70180 | +0.00471 `[-0.00130,+0.01205]` | 0.71593 | -0.00942 `[-0.03227,+0.00942]` |
| 20261010 | HCT116 | 0.64119 | 0.64149 | -0.00030 `[-0.00416,+0.00378]` | 0.63445 | +0.00675 `[-0.00959,+0.02644]` |

These are retrospective internal ChEMBL scaffold-CV results, not prospective biological validation or target-binding predictions. HepG2 and HCT116 are cell lines.

## Data and model scope

- The source cohort uses exact IC50 records with relation `=`, non-null pChEMBL, assay-preserving filtering, and a minimum of ten unique molecules per task-assay.
- The aligned representation cohort contains 3,671 molecules. The ten-conformer ensemble has 30,989 retained conformers. Uni-Mol2 CLS vectors are 768-dimensional; Stage 12 appends six label-free ensemble statistics to arithmetic-mean and Boltzmann-mean CLS vectors.
- Outer validation groups Bemis–Murcko scaffolds; inner model selection uses scaffold-grouped folds. Stage 11/12 use a shared scaffold assignment across both cell lines. Metrics are paired on exact source record IDs and uncertainty is bootstrapped over scaffolds.
- Stage 7 selection is restricted to inner-fold OOF data. Stage 12 PCA, model choice, and nonzero fusion-weight selection are fold-local or inner-selected.

## Reproduce

Install the declared Python environment with `uv sync`. The source scripts expect the curated ChEMBL activity table at `data/raw/origindata/chembl_hepg2_hct116_ic50_eq_calibrated.csv`, the aligned 2D feature archive at `results/2d_morgan_rdkit_features.npz`, and the corresponding 3D feature archives described in each protocol. Obtain Uni-Mol/Uni-Mol2 weights from their official distribution and place them under `models/` before representation extraction.

The benchmark inputs, model checkpoints, raw API payloads, generated feature arrays, and row-level OOF predictions are intentionally not included in this code-focused folder. The committed aggregate tables are sufficient to inspect the reported metrics, but reproducing the full run requires restoring the exact input artifacts and environment described by the protocols. Local paths and credentials are not part of the release.

Useful entry points:

- `src/` — data preparation, conformer extraction, matched model training, and artifact validation.
- `experiments/optimization_v1/` — scaffold split generators, Stage 7/12 benchmarks, selectors, and result auditors.
- `results/` — aggregate seed metrics, paired scaffold-bootstrap summaries, and the Stage 12 gate record.
- `reports/resume_bullet.md` — evidence-bounded résumé wording.
