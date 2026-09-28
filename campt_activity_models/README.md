# Camptothecin Activity Modeling Suite

目标：给定一个分子 SMILES，同时预测 `HepG2` 与 `HCT116` 的 `pIC50`。

本目录是喜树碱衍生物的经典机器学习项目，与仓库中的 ChEMBL Uni-Mol2 84M scaffold benchmark 使用不同数据和评估结果；后者见 [`cpt_unimol_project/activity_benchmark/README.md`](../cpt_unimol_project/activity_benchmark/README.md)。

## 方法目录

- `01_scaffold_stacking_ensemble`：脚手架划分 + 指纹/理化描述符 + Stacking 集成回归
- `02_masked_multitask_mlp`：共享编码器 + 双任务头 + 缺失标签掩码损失
- `03_tanimoto_krr_conformal`：Tanimoto 核岭回归 + Conformal 置信区间

三种方法均使用 scaffold split（seed=42），结果汇总见 `results/model_comparison.csv`。

## 测试集结果

| 方法 | 细胞系 | RMSE | MAE | R² | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|
| Scaffold Stacking Ensemble | HepG2 | 0.746 | 0.549 | 0.327 | 0.584 | 0.594 |
| Masked Multi-task MLP | HepG2 | 0.955 | 0.677 | -0.103 | 0.500 | 0.485 |
| Tanimoto KRR + Conformal | HepG2 | 0.755 | 0.532 | 0.311 | 0.564 | 0.620 |
| Scaffold Stacking Ensemble | HCT116 | 0.904 | 0.666 | 0.294 | 0.568 | 0.597 |
| Masked Multi-task MLP | HCT116 | 1.280 | 0.898 | -0.416 | 0.355 | 0.373 |
| Tanimoto KRR + Conformal | HCT116 | 1.040 | 0.744 | 0.064 | 0.409 | 0.457 |

这些指标来自该模型套件自己的固定测试集，不与 Uni-Mol2 ChEMBL scaffold benchmark 横向混算。

## 运行（仓库根目录执行）

```bash
# 依次训练三种方法
python campt_activity_models/01_scaffold_stacking_ensemble/train.py
python campt_activity_models/02_masked_multitask_mlp/train.py
python campt_activity_models/03_tanimoto_krr_conformal/train.py

# 统一推理（输出三种方法对示例分子的预测）
python campt_activity_models/predict_all_methods.py
```

训练完成后，各方法会在各自 `artifacts/`（不纳入 git）输出：

- `metrics.json`
- `*_model.*`
- `*_test_predictions.csv`
