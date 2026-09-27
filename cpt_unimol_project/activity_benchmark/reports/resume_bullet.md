# Resume Wording

- 基于 ChEMBL 细胞 IC50 数据构建 HepG2/HCT116 活性预测流程，为 3,671 个分子生成 ETKDGv3/MMFF94s/UFF 构象并提取 Uni-Mol2 表征；采用 scaffold 分组嵌套交叉验证比较 2D 基线与 3D-aware 模型。
- 在 seed 20260927 的一组外层 scaffold 切分中，3D-aware 候选的 RMSE 较配对 2D 低 0.0083/0.0057（HepG2/HCT116）；另一预设 seed 未复现，后续多轮比较未形成稳定优势，bootstrap 区间跨 0。
