# Input Data

The raw ChEMBL API payloads and processed molecule/assay tables are not committed. Restore the curated source table at:

`data/raw/origindata/chembl_hepg2_hct116_ic50_eq_calibrated.csv`

The table should retain the source record identifier, canonical SMILES, assay identifier, exact IC50 relation/value, and pChEMBL fields used by the protocols. Do not collapse separate assay records into one row before the known-assay evaluation.
