from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "origindata" / "chembl_hepg2_hct116_ic50_eq_calibrated.csv"
OUT = ROOT / "data" / "raw"
AUDIT = ROOT / "data" / "raw" / "strict_ic50_input_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    outputs = {
        "HepG2": OUT / "hepg2_smiles_pIC50.csv",
        "HCT116": OUT / "hct116_smiles_pIC50.csv",
    }
    counts = {cell: {"source_rows": 0, "selected_rows": 0, "unique_smiles": 0} for cell in outputs}
    labels = {cell: [] for cell in outputs}
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cell = (row.get("cell_line") or "").strip()
            if cell not in outputs:
                continue
            counts[cell]["source_rows"] += 1
            if (row.get("standard_type") or "").strip().upper() != "IC50":
                continue
            if (row.get("standard_relation") or "").strip() != "=":
                continue
            smiles = (row.get("canonical_smiles") or "").strip()
            label = (row.get("pchembl_value_final") or "").strip()
            if not smiles or not label:
                continue
            try:
                value = float(label)
            except ValueError:
                continue
            labels[cell].append((smiles, value))

    for cell, path in outputs.items():
        rows = sorted(labels[cell], key=lambda item: (item[0], item[1]))
        counts[cell]["selected_rows"] = len(rows)
        counts[cell]["unique_smiles"] = len({smiles for smiles, _ in rows})
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["canonical_smiles", "pIC50"])
            writer.writerows(rows)

    audit = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "selection": "standard_type=IC50 and standard_relation='='; pIC50 from pchembl_value_final",
        "outputs": {
            cell: {
                **counts[cell],
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "pIC50_min": min(value for _, value in labels[cell]),
                "pIC50_max": max(value for _, value in labels[cell]),
            }
            for cell, path in outputs.items()
        },
    }
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
