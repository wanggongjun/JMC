# Stage 12 — Uni-Mol2 conformer-distribution summaries

## Hypothesis

Stage 11 used the arithmetic-mean Uni-Mol2 conformer embedding plus a fold-local low-rank projection and did not beat both 2D references on both cell lines. Stage 12 asks a narrower question: does compact within-molecule conformer dispersion carry signal that a pooled mean loses? The new feature view appends six label-free summaries of the covariance spectrum of the ten Uni-Mol2 conformer CLS vectors to the arithmetic and force-field Boltzmann means. This tests an ensemble-distribution hypothesis without training a large attention network on a small cohort.

The 2026 conformer-statistics study uses first- and second-order ensemble summaries and reports selective gains on solvation endpoints; cell-based drug activity was not one of its targets, so this is a method hypothesis only. Axelrod & Gomez-Bombarelli found that conformer-attention models can identify informative conformers, while additional conformers did not consistently improve small drug-activity tasks. Together these findings justify testing compact ensemble statistics under conservative fusion; they do not promise a win.

## Frozen data and features

- Source rows and activity filters are the same assay-aware ChEMBL records as Stage 10, including exact IC50 relation, non-null pChEMBL, at least ten unique molecules per task-assay, no assay/molecule record collapsing, and a matched known-assay estimand.
- Input conformers and representations come from `results/unimol2_84m_10conf_pooled_cls.npz` (3,671 molecules; 30,989 retained conformers; frozen Uni-Mol2 84M; 10-conformer arithmetic and MMFF/UFF Boltzmann CLS vectors).
- For each molecule, the six extra statistics are `log1p(covariance_trace)`, `log1p(covariance_frobenius_norm)`, top-eigenvalue fraction, effective rank, normalized spectral entropy, and Boltzmann effective conformer count. The covariance is computed from the retained per-conformer CLS vectors. Two single-conformer fallback rows have missing force-field energy and use uniform weight for that conformer; this is explicitly recorded. No labels enter feature generation.
- The frozen 1,542-column 3D feature archive is `results/unimol2_ensemble_distribution_features.npz`; its builder recomputes and verifies the Boltzmann pooled vectors against the source artifact.

## Models and validation

- The benchmark is the existing nested assay-aware late-fusion implementation: unchanged per-cell-line 2D selector; 3D candidates use fold-local `StandardScaler` + whitened PCA (16/32/64) + RBF-SVR (`C` 0.1 or 1); the 3D-aware family selects a nonzero prediction blend weight `{0.1, 0.25, 0.5}` using exact inner OOF rows. PCA and blend selection are confined to the current outer-training split.
- A second established per-cell-line 2D selector (SVR, HistGradientBoosting, ExtraTrees and Morgan-Tanimoto KRR) is run on the exact same outer manifests and source rows as a stronger contextual comparator.
- Two new five-fold manifests use seeds `20261009` and `20261010`. Each Bemis-Murcko scaffold has one global fold assignment shared across HepG2 and HCT116. Inner model selection remains four-fold scaffold GroupKFold.
- Frozen manifest SHA-256: seed `20261009` `4e3f72acd611ed7d9c712a6735689533cd030e12ecbcc5377571819a54c5ff7b`; seed `20261010` `af2eca7759b00221a8f8da2c59c26a880f197780f84ebb60f6d7844eb7ceb973`. Both have 870 shared scaffolds, zero cross-task scaffold leakage, and combined record loads of 575/575/576/575/575 and 575/576/575/575/575.
- Frozen code SHA-256: feature builder `build_unimol2_ensemble_distribution_features.py` `212567136990ab541d61942c0d9c14ffa1be067f388a05d7cf6741053f55bf05`; split generator `make_shared_scaffold_fold_manifest.py` `97ae9bdb032a2ff2f3abe398a2da01d768ee39c0aa5b7a15021dc8a4860f9c9f`; benchmark `assay_context_3d_late_fusion_benchmark.py` `12f14df30c81248f0bf428d2fc04d4f7b24414993dbf8451faaf27f98056541c`; independent 2D selector `assay_context_independent_2d_reference.py` `f750036598976655d2d2a2aabb43df7083c25f38d824aedd9b2ce4df70b2666d`.
- Frozen input SHA-256: source rows `78ef6db685909ef1386176636319a4d23a42619f1775f5ff6dac248a0362210d`; Uni-Mol2 pooled conformer archive `ec299ce83bf028a1bf4892f5a8cf119a7f071764fa4b5071f420a8bdd6c2b805`; generated distribution feature archive `c57e5fd195943fc6e4efd11042c82b82124b3ea98374ddfb7c3cc87d76a879d4`.
- Primary scores are known-assay RMSE per task, exact paired OOF records, MAE, R², coverage and paired scaffold-bootstrap 95% intervals.

## Gate

The selected 2D+Uni-Mol2 distribution model must have lower known-assay RMSE than both the matched Stage 10 2D selector and the independent established 2D reference for HepG2 and HCT116 on both fresh manifests. Intervals and effect direction are reported as uncertainty; a one-seed, one-endpoint, or smoke-only win is a fail. This remains internal ChEMBL validation, not prospective biological validation. If this stage fails, record the result and formulate a distinct, bounded experiment rather than quietly relaxing the gate.

## References

- Cheng, Jin & Zhang, “When Does Conformer Geometry Help? Selective Complementarity of 3D Ensemble Statistics and 2D Fingerprints,” ACM-BCB 2026, [DOI: 10.1145/3807503.3819986](https://doi.org/10.1145/3807503.3819986), [arXiv:2606.08825](https://arxiv.org/abs/2606.08825). Its reported advantages are concentrated in solvation properties; covariance summaries are borrowed as a compact feature-design idea, not as evidence for cancer-cell activity.
- Axelrod & Gomez-Bombarelli, “Molecular machine learning with conformer ensembles,” *Machine Learning: Science and Technology* (2023), [arXiv:2012.08452](https://arxiv.org/abs/2012.08452). Their drug-activity experiments motivate conformer-aware modeling and also caution that more conformers can fail to help in small-data settings.
