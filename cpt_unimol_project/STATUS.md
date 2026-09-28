# 历史执行状态快照（2026-03-04）

> 本页只记录旧四阶段项目截至 2026-03-04 的状态，不代表当前项目状态。这里的 `phase2_unimol` 是 Uni-Mol v1（`unimolv1`）；当前 Uni-Mol2 84M 活性基准及完整结果见 [`activity_benchmark/README.md`](activity_benchmark/README.md)。

## ✅ Phase 1：数据深度加工（3D 化）

已完成：
- 构建多任务主表：`data/master_multitask.csv`
- ETKDG 3D 构象：`data/conformers_etkdg.sdf`
- 索引与失败清单：`data/conformer_index.csv`, `data/conformer_failures.csv`

规模：
- 总分子：3749
- 成功构象：3732
- 失败：17

## ✅ Phase 2：Uni-Mol v1 多任务骨架

已完成：
- 脚本：`phase2_unimol/train_unimol_multitask.py`, `phase2_unimol/predict_dual_activity.py`
- 使用 Uni-Mol v1 预训练权重完成 smoke 训练（128 分子，1 epoch；不是完整验证）
- 产物：`phase2_unimol/artifacts_unimol/`（含 5-fold 模型权重）

说明：
- 当前版本 `unimol-tools` 存在 CPU 场景 bug，已在训练脚本中做运行时补丁。
- 全量训练可直接执行（耗时较长）。

## ✅ Phase 3：TopoI 迁移数据准备

已完成：
- 抓取脚本：`phase3_transfer/fetch_top1_chembl.py`
- 标准化脚本：`phase3_transfer/prepare_top1_transfer_data.py`
- 产物：`phase3_transfer/top1_transfer_dataset.csv`

规模：
- Top1 原始活性：635
- 可用迁移样本：502

## ✅ Phase 4：生成器打分接口（首版）

已完成：
- 脚本：`phase4_generator_guidance/score_generated_molecules.py`
- 产物示例：`phase4_generator_guidance/generated_ranked.csv`

当前逻辑：
- 输入生成分子 SMILES
- 输出 HepG2/HCT116 双任务预测与 joint score

## 下一步建议（按顺序）

1. 在全量 3749 数据上执行 Phase 2 正式训练（建议 epochs 20~40）。
2. 用 `top1_transfer_dataset.csv` 进行第二阶段迁移微调（冻结底层层数从 6 调到 8~10 做对比）。
3. 接入你的 2 条 CPT 实验数据做 in-context 校准（最后层线性重标定）。
4. 在 Phase 4 中加入不确定性罚项，形成 `score = w1*hep + w2*hct - λ*uncertainty`。
