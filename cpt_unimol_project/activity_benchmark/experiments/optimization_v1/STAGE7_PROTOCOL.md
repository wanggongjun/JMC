# Stage 7: validation-selected Uni-Mol2 3D-bearing family

Protocol frozen before opening outer OOF metrics for the two repeat runs on 2026-09-25.

## Rationale

The exploratory exact-assay benchmark showed small, task-dependent point improvements from (a) Uni-Mol2 CLS plus RDKit ensemble geometry alone and (b) the same 3D representation concatenated with the 2D fingerprint/descriptors. Neither fixed family improved both cell-line endpoints, so choosing one after seeing its outer score would be invalid. This stage tests a single nested algorithm that chooses between the two 3D-bearing families using only inner OOF validation.

## Frozen evaluation

- Source cohort, assay filter (`IC50`, relation `=`), assay-preserving aggregation, minimum assay size (10 molecules), feature files, known-assay estimand, and model candidate grids remain as in `STAGE2_PROTOCOL.md`.
- Outer folds are the two independently generated scaffold manifests `repeated_scaffold/folds_seed_20260926.csv` and `folds_seed_20260927.csv`.
- For each task and outer fold, select the minimum-inner-OOF-RMSE candidate from the union of assay-centered Uni-Mol2+RDKit-3D-only and 2D+Uni-Mol2+RDKit-3D candidates. Exact ties favor the standalone 3D family. The 2D comparator is separately selected by its own inner OOF RMSE.
- Pair predictions only on exact same assay/molecule records whose assay occurs in the respective outer training partition. Scaffold bootstrap is over paired Bemis-Murcko groups.
- Primary measure: pooled known-assay RMSE. Report endpoint, fold, scaffold-bootstrap interval, selected-family counts, and assay coverage.
- The outer prediction file is never used to choose the family or hyperparameters. The selector writes the inner family selection table before loading outer predictions.

## Pass rule

The 3D-bearing family must have lower pooled known-assay RMSE than the matched 2D model for **both** cell lines on **both** scaffold manifests. A confidence interval crossing zero must be reported as uncertainty; it does not erase the independent-repeat requirement. This is an internal known-assay screening estimate, not an external or prospective validation.
