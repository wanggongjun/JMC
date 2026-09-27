"""Audit source assay diversity and repeated ChEMBL molecule labels."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/origindata/chembl_hepg2_hct116_ic50_eq_calibrated.csv"
OUT = Path(__file__).resolve().parent


def main():
    df = pd.read_csv(SOURCE, encoding="utf-8-sig")
    assay = (df.groupby(["cell_line", "assay_chembl_id"], dropna=False)
             .agg(records=("activity_id", "size"), unique_molecules=("canonical_smiles", "nunique"),
                  mean_pIC50=("pchembl_value_final", "mean"), sd_pIC50=("pchembl_value_final", "std"))
             .reset_index().sort_values(["cell_line", "records"], ascending=[True, False]))
    assay.to_csv(OUT / "assay_counts.csv", index=False)
    repeated = (df.groupby(["cell_line", "canonical_smiles"], dropna=False)
                .agg(n_observations=("activity_id", "size"), mean_pIC50=("pchembl_value_final", "mean"),
                     sd_pIC50=("pchembl_value_final", "std"))
                .reset_index())
    repeated = repeated.loc[repeated.n_observations > 1].copy()
    repeated.to_csv(OUT / "repeated_molecule_labels.csv", index=False)
    summary = {
        "source_rows": int(len(df)),
        "source_assay_ids": int(df.assay_chembl_id.nunique()),
        "source_unique_molecules": int(df.canonical_smiles.nunique()),
        "by_cell_line": {},
        "repeated_cell_line_molecule_pairs": int(len(repeated)),
        "repeated_pair_observations": int(repeated.n_observations.sum()),
        "repeated_pair_pIC50_sd_mean": float(repeated.sd_pIC50.mean()),
        "repeated_pair_pIC50_sd_median": float(repeated.sd_pIC50.median()),
        "interpretation": "Repeated molecule labels may come from different assays and are not assumed to be technical replicates. Their dispersion is a data heterogeneity diagnostic, not a pure assay error estimate.",
    }
    for cell, part in df.groupby("cell_line"):
        summary["by_cell_line"][cell] = {
            "records": int(part.activity_id.nunique()),
            "assays": int(part.assay_chembl_id.nunique()),
            "unique_molecules": int(part.canonical_smiles.nunique()),
        }
    (OUT / "assay_heterogeneity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
