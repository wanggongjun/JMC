# CPT 活性预测项目（四阶段执行）

迁移到新电脑或交接给 AI 时，请先看：`README_NEW_MACHINE.md`。

本目录是独立于其他实验的全新项目实现，按你给出的路线图组织：

1. `phase1_3d`：将 2D SMILES 转成 3D 构象（RDKit ETKDG）
2. `phase2_unimol`：Uni-Mol 风格多任务建模（HepG2/HCT116 双头）
3. `phase3_transfer`：TopoI 迁移学习 + CPT 极小样本校验
4. `phase4_generator_guidance`：面向分子生成器的梯度指导接口

## 当前已实现

- Phase 1 全流程可运行：
  - `phase1_3d/build_master_table.py`
  - `phase1_3d/generate_conformers.py`
- 运行后输出：
  - `data/master_multitask.csv`
  - `data/conformers_etkdg.sdf`
  - `data/conformer_index.csv`
  - `data/conformer_failures.csv`

## 一键运行第一阶段

```powershell
D:/kimi2.5program/JMC/.venv/Scripts/python.exe cpt_unimol_project/phase1_3d/build_master_table.py
D:/kimi2.5program/JMC/.venv/Scripts/python.exe cpt_unimol_project/phase1_3d/generate_conformers.py
```

## 数据来源

- 输入训练标签：
  - `alldata/hepg2_smiles_pIC50.csv`
  - `alldata/hct116_smiles_pIC50.csv`
- 原始备份仍保留在：`alldata/origindata/`

## 后续 Uni-Mol2 scaffold benchmark

2026 年后续评估代码、Stage 7 单次初步改善与跨 seed 复验、Stage 12 最终结果和适用边界说明见 [`activity_benchmark/`](activity_benchmark/README.md)。该项目未证明 3D 表征稳定优于 2D 基线。
