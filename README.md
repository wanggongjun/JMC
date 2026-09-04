# CPT 生物活性预测项目

给定喜树碱（Camptothecin, CPT）衍生物的 SMILES，预测其在 **HepG2** 与 **HCT116** 两个细胞系上的 pIC50。

项目包含两条技术路线：

| 目录 | 路线 | 状态 |
| --- | --- | --- |
| `campt_activity_models/` | 经典机器学习三套模型（主成果，可完整复现） | ✅ 已训练并评估 |
| `cpt_unimol_project/` | Uni-Mol 3D 深度学习多任务管线 | 开发中（见其 README） |

## 方法（campt_activity_models）

1. `01_scaffold_stacking_ensemble`：骨架（scaffold）划分 + RDKit 指纹/理化描述符 + Stacking 集成回归（RF/ET/GBR + Ridge）
2. `02_masked_multitask_mlp`：共享编码器 + HepG2/HCT116 双任务头 + 缺失标签掩码（PyTorch MLP）
3. `03_tanimoto_krr_conformal`：Tanimoto 核岭回归 + conformal 置信区间

统一按 scaffold split（seed=42）评估：80% 训练 / 10% 验证 / 10% 测试。

## 测试集结果

### HepG2

| 方法 | RMSE | MAE | R² | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scaffold Stacking Ensemble | 0.746 | 0.549 | 0.327 | 0.584 | 0.594 |
| Masked Multi-task MLP | 0.955 | 0.677 | -0.103 | 0.500 | 0.485 |
| Tanimoto KRR + Conformal | 0.755 | 0.532 | 0.311 | 0.564 | 0.620 |

### HCT116

| 方法 | RMSE | MAE | R² | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scaffold Stacking Ensemble | 0.904 | 0.666 | 0.294 | 0.568 | 0.597 |
| Masked Multi-task MLP | 1.280 | 0.898 | -0.416 | 0.355 | 0.373 |
| Tanimoto KRR + Conformal | 1.040 | 0.744 | 0.064 | 0.409 | 0.457 |

完整逐条指标见 `campt_activity_models/results/model_comparison.csv`。

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 训练三套经典模型（结果写入各方法 artifacts/ 目录）
python campt_activity_models/01_scaffold_stacking_ensemble/train.py
python campt_activity_models/02_masked_multitask_mlp/train.py
python campt_activity_models/03_tanimoto_krr_conformal/train.py

# 用所有方法预测一个新 SMILES
python campt_activity_models/predict_all_methods.py
```

预测接口示例：

```python
import sys
sys.path.append("campt_activity_models")
from predict_all_methods import main

print(main("CC1=C2C(=O)OC3(CC)C(=O)OCC3C2=NC4=CC=CC=C14"))
```

## 目录结构

```text
.
├── campt_activity_models/   # 经典机器学习主成果（代码 + 评估结果）
│   ├── 01_scaffold_stacking_ensemble/
│   ├── 02_masked_multitask_mlp/
│   ├── 03_tanimoto_krr_conformal/
│   ├── common/              # 公共数据处理/评估工具
│   └── results/             # 模型对比汇总
├── cpt_unimol_project/      # Uni-Mol 深度学习管线（见 README_NEW_MACHINE.md）
├── alldata/                 # 训练标签（来自 ChEMBL/PubChem 公开数据）
└── PANCANCER_ANOVA_*.csv    # 原始药敏汇总（MTEGDRP 数据整理）
```

## 说明与限制

- 训练好的模型权重体积较大，未纳入本仓库；本地训练产物位于各方法 `artifacts/`（已被 .gitignore 排除）。已训练权重如需复用，请留意 GitHub Releases。
- 训练标签的构建/清洗脚本见 `alldata/origindata/`，数据处理细节可完全复现。
- 数据来源为公开数据库（ChEMBL / PubChem / MTEGDRP）；不含未发表实验数据。
- 所有经典模型固定 `seed=42`，同一环境重复运行结果一致。
