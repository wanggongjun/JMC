from pathlib import Path
import json
import hashlib
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_task(path: Path, task: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"canonical_smiles": "smiles", "pIC50": task})[["smiles", task]]
    df[task] = pd.to_numeric(df[task], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["smiles", task]).copy()
    valid = df["smiles"].map(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    invalid = int((~valid).sum())
    df = df[valid].copy()
    n_before_group = len(df)
    df = df.groupby("smiles", as_index=False)[task].mean()
    return df, {
        "file": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "raw_rows": before,
        "dropped_missing": before - n_before_group - invalid,
        "invalid_smiles": invalid,
        "valid_rows_before_aggregation": n_before_group,
        "unique_smiles_after_mean_aggregation": int(len(df)),
        "duplicate_rows_aggregated": n_before_group - int(len(df)),
        "label_min": float(df[task].min()),
        "label_max": float(df[task].max()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hep, hep_audit = load_task(RAW / "hepg2_smiles_pIC50.csv", "hepg2_pIC50")
    hct, hct_audit = load_task(RAW / "hct116_smiles_pIC50.csv", "hct116_pIC50")
    master = hep.merge(hct, how="outer", on="smiles", validate="one_to_one")
    master["n_tasks_observed"] = master[["hepg2_pIC50", "hct116_pIC50"]].notna().sum(axis=1)
    master = master.sort_values("smiles").reset_index(drop=True)
    master.to_csv(OUT / "master_multitask.csv", index=False, encoding="utf-8-sig")
    audit = {
        "label_definition": "pIC50 calculated or curated for exact IC50 records with standard_relation='='; duplicated canonical SMILES are averaged per cell line.",
        "input_audits": {"hepg2": hep_audit, "hct116": hct_audit},
        "master_rows": int(len(master)),
        "hepg2_labeled": int(master.hepg2_pIC50.notna().sum()),
        "hct116_labeled": int(master.hct116_pIC50.notna().sum()),
        "both_labeled": int((master.n_tasks_observed == 2).sum()),
        "scaffold_definition": "RDKit Bemis-Murcko scaffold, chirality excluded; empty scaffolds form one shared group.",
    }
    (OUT / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
