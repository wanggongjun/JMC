# Camptothecin Activity Modeling Suite

目标：给定一个分子 SMILES，同时预测 `HepG2` 与 `HCT116` 的 `pIC50`。

## 方法目录

- `01_scaffold_stacking_ensemble`：脚手架划分 + 指纹/理化描述符 + Stacking 集成回归
- `02_masked_multitask_mlp`：共享编码器 + 双任务头 + 缺失标签掩码损失
- `03_tanimoto_krr_conformal`：Tanimoto 核岭回归 + Conformal 置信区间

三种方法均使用 scaffold split（seed=42），结果汇总见 `results/model_comparison.csv`。

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
