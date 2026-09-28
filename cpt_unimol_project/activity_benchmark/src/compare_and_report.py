from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "matched_model_metrics.json"
DATA_AUDIT = ROOT / "data" / "processed" / "modeling_data_audit.json"
SPLIT_AUDIT = ROOT / "results" / "repeated_split_audit.json"
PRED_AUDIT = ROOT / "results" / "matched_test_set_audit.json"
SPLITS = ROOT / "results" / "repeated_scaffold_splits.csv"
OUT = ROOT / "results" / "model_comparison.csv"
SUMMARY = ROOT / "results" / "model_summary.csv"
REPORT = ROOT / "reports" / "final_report.md"
RESUME = ROOT / "reports" / "resume_bullet.md"
METRIC_KEYS = ("rmse", "mae", "r2", "pearson", "spearman")


def main() -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    data_audit = json.loads(DATA_AUDIT.read_text(encoding="utf-8"))
    split_audit = json.loads(SPLIT_AUDIT.read_text(encoding="utf-8"))
    pred_audit = json.loads(PRED_AUDIT.read_text(encoding="utf-8"))
    if not pred_audit["same_test_smiles_for_2d_and_3d_within_each_seed"]:
        raise RuntimeError("2D and 3D test sets are not identical within each seed")
    seeds = [int(seed) for seed in split_audit["seeds"]]
    if len(seeds) < 2 or sorted(set(seeds)) != sorted(seeds):
        raise RuntimeError("Repeated scaffold evaluation needs at least two distinct seeds")
    if sorted(pred_audit["seeds"]) != sorted(seeds):
        raise RuntimeError("Prediction audit seeds do not match split seeds")
    if any(value != 0 for audit in split_audit["audits"].values() for task in audit["scaffold_overlap"].values() for value in task.values()):
        raise RuntimeError("Scaffold overlap detected")
    if any(not math.isfinite(float(row[key])) for row in metrics for key in METRIC_KEYS):
        raise RuntimeError("Non-finite metric in results")

    splits = pd.read_csv(SPLITS, encoding="utf-8-sig")
    expected = {(int(seed), task, feature) for seed in seeds for task in ("hepg2_pIC50", "hct116_pIC50") for feature in ("2d_morgan_rdkit", "3d_unimol_v1")}
    observed = {(int(row["seed"]), row["task"], row["feature_set"]) for row in metrics}
    if observed != expected:
        raise RuntimeError("Metric rows do not cover every seed/task/feature combination exactly once")
    for (seed, task), group in splits.groupby(["seed", "task"]):
        if group.smiles.duplicated().any():
            raise RuntimeError(f"Duplicate split assignment for seed={seed}, task={task}")
        scaffold_sets = {name: set(part.scaffold) for name, part in group.groupby("split")}
        if any(scaffold_sets[a] & scaffold_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise RuntimeError(f"Scaffold overlap in saved splits: seed={seed}, task={task}")

    table = pd.DataFrame(metrics)
    table["model"] = table["feature_set"].map({
        "2d_morgan_rdkit": "2D Morgan + RDKit descriptors",
        "3d_unimol_v1": "Uni-Mol v1 3D CLS embedding",
    })
    table.to_csv(OUT, index=False, encoding="utf-8-sig")
    grouped = table.groupby(["task", "feature_set", "model"], sort=False)
    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        n_test_mean=("n_test", "mean"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        pearson_mean=("pearson", "mean"), pearson_std=("pearson", "std"),
        spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
    ).reset_index()
    if (summary.n_seeds != len(seeds)).any():
        raise RuntimeError("At least one model summary is missing seed results")
    summary.to_csv(SUMMARY, index=False, encoding="utf-8-sig")

    by_key = {(row.task, row.feature_set): row for row in summary.itertuples(index=False)}
    lines = [
        "# HepG2/HCT116 双细胞系活性预测：Uni-Mol v1 3D 与 2D 基线",
        "",
        "## 数据",
        "",
        "只使用 ChEMBL 精确 IC50（standard_type=IC50 且 standard_relation='='）记录；canonical SMILES 重复值按细胞系取均值。",
        f"原始清洗集合 {data_audit['master_rows']} 个分子；其中 {data_audit['rows_with_cached_3d_conformer']} 个有可用缓存构象，用于 2D/3D 同样本对照；{data_audit['rows_excluded_no_usable_conformer']} 个无可用构象样本排除。",
        f"构象状态：成功优化 {data_audit['included_conformer_status_counts'].get('ok', 0)} 个，UFF 未收敛但仍有有效 3D 坐标 {data_audit['included_conformer_status_counts'].get('uff_not_converged', 0)} 个；未使用状态为 {data_audit['excluded_status_counts']}。",
        f"建模子集标签数：HepG2 {data_audit['hepg2_rows']}，HCT116 {data_audit['hct116_rows']}；仅 {data_audit['both_tasks_rows']} 个分子在两细胞系都有标签，因此按细胞系分别训练和报告。",
        "",
        "## 方法",
        "",
        "- 2D：Morgan radius=2、2048-bit 指纹 + 9 项 RDKit 理化描述符。",
        "- 3D：Uni-Mol v1 all-H 官方预训练权重提取 512 维 CLS 表征；使用现有 RDKit ETKDG 坐标，不做端到端微调。",
        "- 采用相同 ExtraTrees 候选集（120 棵树、叶节点最小样本数 2/8、sqrt 特征采样），仅按各 seed 的 validation RMSE 选择。",
        f"- 对 5 组 seed={', '.join(map(str, seeds))} 独立做约 80/10/10 Bemis–Murcko scaffold holdout；每轮在 train+validation 拟合后仅评估对应 test。",
        "- 每个 seed 内 2D/3D 共享完全相同的测试 SMILES；每次拆分均审计 scaffold 交集为 0。",
        "",
        "## 测试指标（跨拆分均值 ± 标准差）",
        "",
        "| 细胞系 | 表征 | 平均测试 n | RMSE | MAE | R² | Pearson | Spearman |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"hepg2_pIC50": "HepG2", "hct116_pIC50": "HCT116", "2d_morgan_rdkit": "2D Morgan + RDKit", "3d_unimol_v1": "Uni-Mol v1 3D CLS"}
    for task in ("hepg2_pIC50", "hct116_pIC50"):
        for feature in ("2d_morgan_rdkit", "3d_unimol_v1"):
            row = by_key[(task, feature)]
            lines.append(
                f"| {labels[task]} | {labels[feature]} | {row.n_test_mean:.1f} | "
                f"{row.rmse_mean:.3f} ± {row.rmse_std:.3f} | {row.mae_mean:.3f} ± {row.mae_std:.3f} | "
                f"{row.r2_mean:.3f} ± {row.r2_std:.3f} | {row.pearson_mean:.3f} ± {row.pearson_std:.3f} | "
                f"{row.spearman_mean:.3f} ± {row.spearman_std:.3f} |"
            )
    lines += ["", "完整逐 seed、逐分子的指标和预测保存在 `results/model_comparison.csv` 与 `results/predictions/`。", "", "## 结果解释与限制", ""]
    deltas = {}
    for task in ("hepg2_pIC50", "hct116_pIC50"):
        two = by_key[(task, "2d_morgan_rdkit")]
        three = by_key[(task, "3d_unimol_v1")]
        delta = three.rmse_mean - two.rmse_mean
        deltas[task] = delta
        comparison = "较高" if delta > 0 else "较低" if delta < 0 else "相同"
        lines.append(f"- {labels[task]}：Uni-Mol v1 3D 的跨拆分平均 RMSE 比同样本 2D 基线{comparison} {abs(delta):.3f}。")
    lines += [
        "",
        "5 个 seed 反映对 scaffold 分组随机性的敏感程度，不是独立外部验证，也不是置信区间；测试集之间可能含有重复分子。UFF 未收敛构象仍有静态 3D 坐标，其影响没有单独消融。结果描述细胞系体外活性数据，不等于特定靶点结合或临床疗效。",
        "",
        "原先全量 2D stacking 指标保存在 `baselines/2d_existing/results/model_comparison.csv`；其样本未按 3D 构象可用性筛选，不与本报告的同样本指标直接横比。",
        "",
        "复现命令见项目根目录 `README.md`。输入表、拆分、Uni-Mol 权重哈希、表征、预测、指标及模型文件均随项目保存。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    hep3d = by_key[("hepg2_pIC50", "3d_unimol_v1")]
    hct3d = by_key[("hct116_pIC50", "3d_unimol_v1")]
    hep2d = by_key[("hepg2_pIC50", "2d_morgan_rdkit")]
    hct2d = by_key[("hct116_pIC50", "2d_morgan_rdkit")]
    bullet = (
        f"基于 ChEMBL 严格 IC50 数据构建 HepG2/HCT116 双细胞系活性预测集，"
        f"在 {len(seeds)} 组 Bemis–Murcko scaffold holdout 上，同样本比较 Uni-Mol v1 预训练 3D 表征与 2D Morgan/RDKit 基线；"
        f"3D 平均测试 RMSE 为 {hep3d.rmse_mean:.3f}±{hep3d.rmse_std:.3f}/{hct3d.rmse_mean:.3f}±{hct3d.rmse_std:.3f}，"
        f"2D 基线为 {hep2d.rmse_mean:.3f}±{hep2d.rmse_std:.3f}/{hct2d.rmse_mean:.3f}±{hct2d.rmse_std:.3f}（HepG2/HCT116）。"
    )
    RESUME.write_text("## 简历建议（按实测结果填写）\n\n" + bullet + "\n", encoding="utf-8")
    print(f"saved {OUT}\nsaved {SUMMARY}\nsaved {REPORT}\nsaved {RESUME}\nseeds={seeds}")


if __name__ == "__main__":
    main()
