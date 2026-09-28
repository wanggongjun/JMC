# CPT Activity Project Archive and Uni-Mol2 Benchmark

This folder contains an earlier CPT-focused modeling pipeline and the newer, separate ChEMBL cell-activity benchmark. The repository-level [`README.md`](../README.md) is the landing page; the current benchmark details are in [`activity_benchmark/README.md`](activity_benchmark/README.md).

## Current benchmark

`activity_benchmark/` compares frozen Uni-Mol2 84M conformer representations with matched 2D models on HepG2 and HCT116 ChEMBL IC50 records. Its Stage 7/12 methods, aggregate metrics, and validation limits are documented there. The latest evaluation did not establish a reproducible 3D advantage.

## Earlier CPT pipeline

The phase folders below preserve the original four-stage project structure. In particular, `phase2_unimol/` uses Uni-Mol v1 (`model_name="unimolv1"`); it is not the Uni-Mol2 implementation or the source of the current benchmark metrics. The dated setup and status notes for that older pipeline are [`README_NEW_MACHINE.md`](README_NEW_MACHINE.md) and [`STATUS.md`](STATUS.md).

The original phases are:

1. `phase1_3d`: RDKit ETKDG conformer generation.
2. `phase2_unimol`: the earlier Uni-Mol v1 multitask pipeline.
3. `phase3_transfer`: TopoI transfer-data preparation and CPT small-sample work.
4. `phase4_generator_guidance`: an initial scoring interface for generated molecules.

These directories contain separate historical work; consult each phase's scripts and dated status before treating it as a current, validated result.
