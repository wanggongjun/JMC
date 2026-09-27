# Stage 2: complementary 3D representation test

Date fixed: 2026-09-25

## Hypothesis

Uni-Mol2 ensemble embeddings encode learned local molecular environments, while the RDKit ensemble descriptors summarize global conformational shape and accessible surface. Their concatenation may give a more useful 3D-only representation, or complementary information when paired with the existing 2D model.

## Data and primary estimand

- Source: exact ChEMBL IC50 (`standard_relation = "="`) pChEMBL records in `chembl_hepg2_hct116_ic50_eq_calibrated.csv`.
- Cohort: assays with at least 10 distinct molecules after requiring both 2D and 3D feature coverage; assay-molecule duplicates are retained as source records and all records for a molecule stay in the same scaffold fold.
- Primary estimand: known-assay screening. For a test record, its assay must be represented in that model's training partition; the model predicts a training-only assay mean plus a molecular residual. Unknown-assay fallback is reported separately.
- Primary metric: RMSE. Secondary: MAE, R², and paired scaffold bootstrap interval for candidate-minus-2D RMSE.
- Leakage control: frozen molecule-level Bemis-Murcko outer folds, nested scaffold-only model selection, fold-local feature scaling, and training-only assay means.

## Comparison

Use `assay_context_3d_benchmark.py` without changing its model-selection grid or primary data filters:

1. 2D-only reference;
2. concatenated Uni-Mol2 CLS + RDKit rich 3D geometry as a 3D-only model;
3. 2D + concatenated 3D as a clearly labeled hybrid;
4. assay-mean-only reference.

The feature artifact is generated reproducibly by `combine_unimol2_rdkit3d_features.py`; exact canonical-SMILES coverage and source hashes are recorded in its metadata file. Do not choose a favorable task, fold, split seed, metric, or assay subset after reading outer-test outcomes. The original frozen split is exploratory for this representation; if it is promising, require both pre-generated independent scaffold manifests (`folds_seed_20260926.csv` and `folds_seed_20260927.csv`) before considering a résumé claim.

## Decision rule

The activity-prediction delivery gate remains unchanged: the same 3D-only or clearly labeled 2D+3D method must have a lower primary RMSE than the matched 2D-only baseline on both HepG2 and HCT116, and the direction must persist on both independent scaffold manifests. Report every endpoint, delta, fold result, paired scaffold interval, and assay-support limitation. If this fails, do not promote a favorable fold or secondary metric as a pass.
