from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import math
import re
import sys
from typing import Dict, List, Optional, Tuple

from unimol_tools import MolPredict


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpt_unimol_project.phase2_unimol.predict_dual_activity import _extract_dual_values, resolve_model_dir


@dataclass
class PredictionRecord:
    compound: str
    smiles: str
    hepg2_pic50: Optional[float]
    hct116_pic50: Optional[float]
    error: Optional[str]


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip()).upper()


def ic50_um_to_pic50(value_um: Optional[float]) -> Optional[float]:
    if value_um is None or value_um <= 0:
        return None
    return 6.0 - math.log10(value_um)


def pic50_to_ic50_um(value_pic50: Optional[float]) -> Optional[float]:
    if value_pic50 is None:
        return None
    return 10 ** (6.0 - float(value_pic50))


def parse_smiles_file(path: Path) -> Dict[str, Tuple[str, str]]:
    smiles_map: Dict[str, Tuple[str, str]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" not in line:
            continue
        compound, smiles = line.split("\t", 1)
        compound = compound.strip()
        smiles = smiles.strip()
        if compound and smiles:
            smiles_map[normalize_name(compound)] = (compound, smiles)
    return smiles_map


def parse_table1_ic50(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    data_lines: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        data_lines.append(line)

    reader = csv.DictReader(data_lines, delimiter="\t")
    for row in reader:
        compound = (row.get("Compound") or "").strip()
        if not compound:
            continue
        rows.append(
            {
                "compound": compound,
                "a549_ic50_um": float(row["A549_IC50"]),
                "hct116_ic50_um": float(row["HCT-116_IC50"]),
            }
        )
    return rows


def parse_table2_hepg2(path: Path) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(B7|SN-38)\s+", line)
        if not match:
            continue
        target = match.group(1)
        values = [float(v) for v in re.findall(r"\d+\.\d+", line)]
        if len(values) >= 8:
            means = values[0::2]
            if len(means) >= 4:
                results[target] = means[3]
    return results


def run_predictions(smiles_map: Dict[str, Tuple[str, str]]) -> Dict[str, PredictionRecord]:
    predictions: Dict[str, PredictionRecord] = {}
    predictor = MolPredict(load_model=str(resolve_model_dir()))
    for normalized_name, (compound, smiles) in smiles_map.items():
        try:
            raw_output = predictor.predict({'SMILES': [smiles]})
            values = _extract_dual_values(raw_output)
            predictions[normalized_name] = PredictionRecord(
                compound=compound,
                smiles=smiles,
                hepg2_pic50=values.get("hepg2_pIC50"),
                hct116_pic50=values.get("hct116_pIC50"),
                error=None,
            )
        except Exception as exc:
            predictions[normalized_name] = PredictionRecord(
                compound=compound,
                smiles=smiles,
                hepg2_pic50=None,
                hct116_pic50=None,
                error=str(exc),
            )
    return predictions


def summarize_errors(values: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None
    mae = sum(abs(v) for v in values) / len(values)
    rmse = math.sqrt(sum(v * v for v in values) / len(values))
    max_abs = max(abs(v) for v in values)
    return mae, rmse, max_abs


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def build_report(
    literature_rows: List[Dict[str, object]],
    hepg2_rows: Dict[str, float],
    predictions: Dict[str, PredictionRecord],
    source_dir: Path,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_dir = resolve_model_dir()
    lines: List[str] = []
    lines.append("JMC 模型与文献活性对比报告")
    lines.append(f"生成时间: {now}")
    lines.append(f"数据目录: {source_dir}")
    lines.append(f"模型目录: {model_dir}")
    lines.append("")
    lines.append("说明:")
    lines.append("- 模型输出为 HepG2 与 HCT116 的 pIC50。")
    lines.append("- 文献 Table 1 提供 A549 与 HCT-116 的 IC50(μM)，其中只有 HCT-116 可与模型直接对比。")
    lines.append("- 文献 Table 2 提供 B7 与 SN-38 在 HepG2 的 IC50(μM)，可与模型 HepG2 输出直接对比。")
    lines.append("- 误差主尺度使用 pIC50: signed_error = predicted_pIC50 - literature_pIC50。")
    lines.append("- 另给出 IC50 倍数误差: predicted_IC50 / literature_IC50。")
    lines.append("")

    hct_errors: List[float] = []
    hepg2_errors: List[float] = []
    prediction_failures: List[PredictionRecord] = []

    lines.append("HCT116 对比结果")
    lines.append("compound\tlit_ic50_uM\tlit_pIC50\tpred_pIC50\tpred_ic50_uM\tsigned_err_pIC50\tabs_err_pIC50\tfold_err_ic50")
    for row in literature_rows:
        compound = str(row["compound"])
        record = predictions.get(normalize_name(compound))
        lit_ic50_um = float(row["hct116_ic50_um"])
        lit_pic50 = ic50_um_to_pic50(lit_ic50_um)
        if record is None or record.error:
            err_text = record.error if record else "SMILES missing"
            lines.append(f"{compound}\t{lit_ic50_um:.4f}\t{format_float(lit_pic50)}\tN/A\tN/A\tN/A\tN/A\tN/A ({err_text})")
            if record and record.error:
                prediction_failures.append(record)
            continue
        pred_pic50 = record.hct116_pic50
        pred_ic50_um = pic50_to_ic50_um(pred_pic50)
        signed_err = None if pred_pic50 is None or lit_pic50 is None else pred_pic50 - lit_pic50
        abs_err = None if signed_err is None else abs(signed_err)
        fold_err = None if pred_ic50_um is None else pred_ic50_um / lit_ic50_um
        if signed_err is not None:
            hct_errors.append(signed_err)
        lines.append(
            f"{compound}\t{lit_ic50_um:.4f}\t{format_float(lit_pic50)}\t{format_float(pred_pic50)}\t"
            f"{format_float(pred_ic50_um)}\t{format_float(signed_err)}\t{format_float(abs_err)}\t{format_float(fold_err)}"
        )

    lines.append("")
    lines.append("HepG2 对比结果")
    lines.append("compound\tlit_ic50_uM\tlit_pIC50\tpred_pIC50\tpred_ic50_uM\tsigned_err_pIC50\tabs_err_pIC50\tfold_err_ic50")
    for compound, lit_ic50_um in hepg2_rows.items():
        record = predictions.get(normalize_name(compound))
        lit_pic50 = ic50_um_to_pic50(lit_ic50_um)
        if record is None or record.error:
            err_text = record.error if record else "SMILES missing"
            lines.append(f"{compound}\t{lit_ic50_um:.4f}\t{format_float(lit_pic50)}\tN/A\tN/A\tN/A\tN/A\tN/A ({err_text})")
            if record and record.error:
                prediction_failures.append(record)
            continue
        pred_pic50 = record.hepg2_pic50
        pred_ic50_um = pic50_to_ic50_um(pred_pic50)
        signed_err = None if pred_pic50 is None or lit_pic50 is None else pred_pic50 - lit_pic50
        abs_err = None if signed_err is None else abs(signed_err)
        fold_err = None if pred_ic50_um is None else pred_ic50_um / lit_ic50_um
        if signed_err is not None:
            hepg2_errors.append(signed_err)
        lines.append(
            f"{compound}\t{lit_ic50_um:.4f}\t{format_float(lit_pic50)}\t{format_float(pred_pic50)}\t"
            f"{format_float(pred_ic50_um)}\t{format_float(signed_err)}\t{format_float(abs_err)}\t{format_float(fold_err)}"
        )

    hct_mae, hct_rmse, hct_max = summarize_errors(hct_errors)
    hep_mae, hep_rmse, hep_max = summarize_errors(hepg2_errors)

    lines.append("")
    lines.append("汇总统计")
    lines.append(f"HCT116 可比样本数: {len(hct_errors)}")
    lines.append(f"HCT116 pIC50 MAE: {format_float(hct_mae)}")
    lines.append(f"HCT116 pIC50 RMSE: {format_float(hct_rmse)}")
    lines.append(f"HCT116 最大绝对误差: {format_float(hct_max)}")
    lines.append(f"HepG2 可比样本数: {len(hepg2_errors)}")
    lines.append(f"HepG2 pIC50 MAE: {format_float(hep_mae)}")
    lines.append(f"HepG2 pIC50 RMSE: {format_float(hep_rmse)}")
    lines.append(f"HepG2 最大绝对误差: {format_float(hep_max)}")

    a549_only = [str(row["compound"]) for row in literature_rows]
    lines.append("")
    lines.append("A549 说明")
    lines.append("- 文献中全部化合物都有 A549 IC50，但当前 JMC 模型不输出 A549，因此这部分只能保留文献值，不能计算直接误差。")
    lines.append(f"- 涉及 A549 文献样本数: {len(a549_only)}")

    unique_failures = []
    seen = set()
    for item in prediction_failures:
        key = normalize_name(item.compound)
        if key in seen:
            continue
        seen.add(key)
        unique_failures.append(item)

    lines.append("")
    lines.append("预测失败或不可用项")
    if unique_failures:
        for item in unique_failures:
            lines.append(f"- {item.compound}: {item.error}")
    else:
        lines.append("- 无")

    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) > 1:
        source_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        source_dir = Path(r"C:\Users\danielshark\Desktop\生物活性\ceshiji")

    smiles_path = source_dir / "camptothecin_compounds_smiles.txt"
    ic50_path = source_dir / "camptothecin_compounds_ic50.txt"
    extracted_path = source_dir / "extracted_text.txt"
    output_path = source_dir / "JMC_文献对比报告.txt"

    if not smiles_path.exists():
        raise FileNotFoundError(smiles_path)
    if not ic50_path.exists():
        raise FileNotFoundError(ic50_path)
    if not extracted_path.exists():
        raise FileNotFoundError(extracted_path)

    smiles_map = parse_smiles_file(smiles_path)
    literature_rows = parse_table1_ic50(ic50_path)
    hepg2_rows = parse_table2_hepg2(extracted_path)
    predictions = run_predictions(smiles_map)
    report = build_report(literature_rows, hepg2_rows, predictions, source_dir)
    output_path.write_text(report, encoding="utf-8-sig")

    print(output_path)


if __name__ == "__main__":
    main()