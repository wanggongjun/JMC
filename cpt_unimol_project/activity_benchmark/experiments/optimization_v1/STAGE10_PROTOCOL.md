# Stage 10: nested shrinkage late fusion

## Rationale

Stages 8–9 show that 3D information may help in some folds, but replacing the 2D predictor is unstable. Stage 9 selected a 2D+compressed-3D feature model in 18 of 20 outer folds, yet did not produce a repeated test-set gain. Stage 10 tests a conservative late-fusion hypothesis: preserve the independently selected 2D prediction and add a small, inner-CV-selected contribution from a low-rank Uni-Mol2 predictor.

## Frozen models

- Reuse the exact Stage 8 2D and 10-conformer 3D feature artifacts and data filters: ChEMBL exact-relation IC50, non-null pChEMBL, minimum 10 unique molecules per assay, train-only assay means, and the same known-assay primary estimand.
- The 2D reference remains the unchanged Stage 8 family: its existing 2D vector candidates and Tanimoto-kernel candidates, selected by inner known-assay RMSE.
- The 3D base candidates are fold-local standardized PCA of the 1,558-column Uni-Mol2/RDKit view, retaining 16, 32, or 64 components, followed by RBF-SVR (`epsilon=0.15`, `C` 0.1 or 1). PCA is fit only on the current training fold.
- In each outer-training partition, generate exact-record inner OOF predictions from the independently selected 2D base model and every 3D base candidate. Compare 3D-only predictions and fixed late blends `prediction = (1-w) * prediction_2D + w * prediction_3D`, with nonzero `w` in `{0.1, 0.25, 0.5}`. Select the 3D candidate and weight only by inner known-assay RMSE. The outer model refits the selected 2D and 3D bases on outer training data and applies the frozen blend weight to outer predictions.
- The 3D-aware family cannot select a 2D-only predictor. No test-fold value, outer score, assay subset, or endpoint is used to set blend weights.

## Evaluation gate

- Freeze two new five-fold shuffled Bemis-Murcko scaffold manifests with seeds `20261005` and `20261006` before generating any Stage 10 outer predictions.
- Frozen manifest SHA-256 values: seed `20261005` `e430cd52d890af67127d5da3c3dd3fe1e184ca984406e7834a7d8c61f67e008b`; seed `20261006` `a8c9c3460c4d07f9e0f624dc9ef8455d4c428317f8b9e42ab89f78336cb3ffc2`. Dataset SHA-256: `609f59f3036f25c956f1e885e316ccc9fefe66e9589df9abe99b6f0c8d1bb09c`.
- Frozen implementation: `assay_context_3d_late_fusion_benchmark.py`, SHA-256 `12f14df30c81248f0bf428d2fc04d4f7b24414993dbf8451faaf27f98056541c`. It compiled and completed a one-outer-fold development smoke on an older manifest; that output is marked `smoke_only` and excluded from results.
- Keep nested four-fold scaffold GroupKFold, exact paired OOF records, pooled known-assay RMSE as primary metric, and report overall/known-assay RMSE, MAE, R², fold results, assay coverage, candidate/weight selection counts, and paired scaffold-bootstrap 95% intervals.
- Pass only if the selected 3D-bearing predictor beats the independently selected 2D reference on both HepG2 and HCT116 on both new manifests. An interval crossing zero is reported as uncertainty and does not become evidence of a significant gain. No endpoint, metric, threshold, or split changes after result inspection.
- Do not inspect any final outer summary until both Stage 10 seed jobs finish. If the gate fails, record the result and continue only with a distinct frozen method and fresh manifests.
- A résumé claim must identify the cell-line pIC50 endpoint, ChEMBL data, and known-assay scaffold-CV condition. Internal CV is not prospective biological validation.

## Known limits

The source aggregates heterogeneous assays, the force-field Boltzmann weights are approximations, the Uni-Mol2 encoder remains frozen, and even a pass would require prospective experiments before making a biological validation claim.

## Post-run result

Both frozen Stage 10 manifests completed with exit code 0 and zero outer-molecule overlap. The selected 3D-aware candidate was a nonzero late blend in all 20 task/fold fits. The primary gate failed because HCT116 did not improve consistently: seed 20261005 HepG2 delta `-0.004231` (95% CI `[-0.007187,-0.001245]`), HCT116 `-0.001191` (`[-0.003538,+0.001155]`); seed 20261006 HepG2 `-0.003764` (`[-0.010459,+0.003629]`), HCT116 `+0.001328` (`[-0.002696,+0.005860]`). This is an internal, task-limited signal and does not support a dual-cell-line superiority claim.
