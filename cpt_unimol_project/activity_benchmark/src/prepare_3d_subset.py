from pathlib import Path
import json
import pickle
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "processed" / "master_multitask.csv"
CACHE = ROOT / "data" / "processed" / "conformer_features.pkl"
OUT = ROOT / "data" / "processed" / "modeling_dataset.csv"
AUDIT = ROOT / "data" / "processed" / "modeling_data_audit.json"


def main() -> None:
    master = pd.read_csv(MASTER, encoding="utf-8-sig")
    with CACHE.open("rb") as f:
        entries = pickle.load(f)
    by_smiles = {str(item["smiles"]): item for item in entries}
    usable = {
        smiles: item for smiles, item in by_smiles.items()
        if item.get("atoms") is not None and item.get("coords") is not None
        and len(item["atoms"]) == len(item["coords"])
    }
    modeling = master.loc[master.smiles.astype(str).isin(usable)].copy()
    modeling["conformer_status"] = modeling.smiles.astype(str).map(lambda s: by_smiles[s].get("status", "unknown"))
    modeling = modeling.sort_values("smiles").reset_index(drop=True)
    modeling.to_csv(OUT, index=False, encoding="utf-8-sig")
    status_counts = {str(k): int(v) for k, v in modeling.conformer_status.value_counts().items()}
    audit = {
        "master_rows": int(len(master)),
        "rows_with_cached_3d_conformer": int(len(modeling)),
        "rows_excluded_no_usable_conformer": int(len(master) - len(modeling)),
        "excluded_status_counts": {str(k): int(v) for k, v in pd.Series([i.get("status", "unknown") for i in entries if i.get("atoms") is None or i.get("coords") is None]).value_counts().items()},
        "included_conformer_status_counts": status_counts,
        "hepg2_rows": int(modeling.hepg2_pIC50.notna().sum()),
        "hct116_rows": int(modeling.hct116_pIC50.notna().sum()),
        "both_tasks_rows": int((modeling.n_tasks_observed == 2).sum()),
        "conformer_source": "existing RDKit ETKDG conformer_features.pkl; coordinates used as cached, without regeneration",
        "note": "UFF non-converged conformers are retained because valid 3D coordinates exist; status is preserved for audit.",
    }
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
