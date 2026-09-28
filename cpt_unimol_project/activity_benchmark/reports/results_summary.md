# Aggregate Results Summary

The Stage 7 and Stage 12 3D representations in this report use frozen Uni-Mol2 84M (`unimolv2`) conformer embeddings. Earlier Uni-Mol v1 experiments are outside these aggregate comparisons.

## Stage 7: validation-selected 3D-bearing model family

The protocol selected between two 3D-bearing candidate families using only inner-fold OOF RMSE. The comparison below uses the paired assay-centered 2D baseline on two fresh scaffold manifests.

| Seed | Cell line | 2D RMSE | 3D-bearing RMSE | ΔRMSE | Paired scaffold-bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| 20260926 | HepG2 | 0.63885 | 0.64754 | +0.00869 | [-0.00430, +0.02051] |
| 20260926 | HCT116 | 0.73257 | 0.75024 | +0.01767 | [+0.00238, +0.03164] |
| 20260927 | HepG2 | 0.66579 | 0.65754 | -0.00825 | [-0.03381, +0.01241] |
| 20260927 | HCT116 | 0.71894 | 0.71326 | -0.00569 | [-0.01866, +0.00833] |

The seed-20260927 point estimates are lower on both endpoints, but the uncertainty intervals cross zero and the direction reverses on seed 20260926. This remains an exploratory, split-sensitive signal.

## Stage 12: conformer-distribution late fusion

Stage 12 uses a 1,542-column feature archive: arithmetic-mean CLS, force-field Boltzmann-mean CLS, and six conformer-dispersion summaries. The table compares the selected 2D+3D candidate with the matched nested 2D model and a stronger independent 2D selector.

| Seed | Cell line | Candidate RMSE | Matched 2D RMSE | Δ vs matched 2D (95% CI) | Independent 2D RMSE | Δ vs independent 2D (95% CI) |
|---|---|---:|---:|---:|---:|---:|
| 20261009 | HepG2 | 0.64094 | 0.64624 | -0.00530 [-0.01408, +0.00294] | 0.64981 | -0.00888 [-0.02591, +0.00824] |
| 20261009 | HCT116 | 0.68079 | 0.68119 | -0.00040 [-0.00573, +0.00425] | 0.69376 | -0.01297 [-0.03059, +0.00446] |
| 20261010 | HepG2 | 0.70651 | 0.70180 | +0.00471 [-0.00130, +0.01205] | 0.71593 | -0.00942 [-0.03227, +0.00942] |
| 20261010 | HCT116 | 0.64119 | 0.64149 | -0.00030 [-0.00416, +0.00378] | 0.63445 | +0.00675 [-0.00959, +0.02644] |

The Stage 12 prespecified gate fails: the candidate does not beat both 2D references on both endpoints and both seeds. All paired bootstrap intervals cross zero. These are internal retrospective scaffold-CV estimates, not prospective assay validation.

## Interpretation

The work demonstrates a complete data-to-evaluation workflow and quantifies a preliminary seed-specific improvement. It does not support a general claim that Uni-Mol2 3D improves cell-activity prediction over the 2D baseline.
