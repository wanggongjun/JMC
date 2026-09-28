# Molecular Activity Prediction Projects

This repository contains two related but separate projects: classic models for camptothecin derivatives and a later ChEMBL cell-line benchmark using Uni-Mol2. Their datasets and results should not be mixed.

## Current 3D benchmark: Uni-Mol2 84M

The maintained comparison is in [`cpt_unimol_project/activity_benchmark/`](cpt_unimol_project/activity_benchmark/README.md). It uses frozen `unimolv2` 84M conformer embeddings and scaffold-grouped evaluation against matched 2D baselines for ChEMBL HepG2 and HCT116 IC50 records.

The aligned cohort contains 3,671 molecules and 30,989 conformers. In one exploratory Stage 7 split (seed `20260927`), RMSE was lower than the paired 2D baseline by 0.00825 for HepG2 and 0.00569 for HCT116. The other prespecified seed reversed direction, and the Stage 12 replication gate did not pass. The evidence does not establish a stable 3D advantage; the full metrics and limitations are documented in the benchmark README.

## Classic camptothecin models

[`campt_activity_models/`](campt_activity_models/README.md) contains three scaffold-split approaches for predicting pIC50 from camptothecin-derivative SMILES: scaffold stacking, a masked multitask MLP, and Tanimoto kernel ridge regression. Its dataset and metrics are separate from the broader ChEMBL benchmark.

## Model-version map

- `cpt_unimol_project/activity_benchmark/`: the headline Stage 7 and Stage 12 comparisons use Uni-Mol2 84M (`model_name="unimolv2"`).
- `cpt_unimol_project/phase2_unimol/`: an earlier Uni-Mol v1 pipeline (`model_name="unimolv1"`), retained as historical work. Its runs are not the Uni-Mol2 benchmark results.
- `cpt_unimol_project/activity_benchmark/src/` and some early experiment scripts also contain Uni-Mol v1 exploratory code. The Stage 7/12 protocols identify the Uni-Mol2 inputs and procedures used for their reported metrics.

## Data and reproduction

The benchmark folder publishes source code, protocols, and aggregate metrics. It does not include the curated ChEMBL input table, pretrained weights, generated feature arrays, or row-level predictions. See its [data, weight, and reproduction notes](cpt_unimol_project/activity_benchmark/README.md#reproduce) before attempting a full rerun. Classic-model setup instructions are in the `campt_activity_models` README.
