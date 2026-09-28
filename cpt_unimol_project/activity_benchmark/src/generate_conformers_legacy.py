from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'processed'


def make_3d_mol(smiles: str, seed: int = 42, max_attempts: int = 3):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, 'invalid_smiles'

    # Speed guard: reduce salt/mixture complexity by keeping the largest fragment.
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda x: x.GetNumHeavyAtoms())

    # Skip extremely large molecules to avoid minute-level stalls in RDKit embedding.
    if mol.GetNumHeavyAtoms() > 70:
        return None, 'skip_too_large'

    mol = Chem.AddHs(mol)

    conf_id = -1
    for i in range(max_attempts):
        # Cap attempts per call to prevent single hard molecules from blocking for minutes.
        conf_id = AllChem.EmbedMolecule(
            mol,
            maxAttempts=20,
            randomSeed=seed + i,
            useRandomCoords=True,
            clearConfs=True,
        )
        if conf_id >= 0:
            break

    if conf_id < 0:
        return None, 'embed_failed'

    try:
        max_iters = 100 if mol.GetNumAtoms() > 90 else 200
        opt_status = AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iters)
        status = 'ok' if opt_status == 0 else 'uff_not_converged'
    except Exception:
        status = 'uff_failed'

    return mol, status


def extract_xyz(mol: Chem.Mol, conf_id: int = 0):
    conf = mol.GetConformer(conf_id)
    coords = []
    atoms = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([float(p.x), float(p.y), float(p.z)])
        atoms.append(atom.GetSymbol())
    return np.array(coords, dtype=np.float32), atoms


def main() -> None:
    in_csv = DATA_DIR / 'master_multitask.csv'
    if not in_csv.exists():
        raise FileNotFoundError(f'master file not found: {in_csv}, run build_master_table.py first')

    df = pd.read_csv(in_csv, encoding='utf-8-sig')

    sdf_path = DATA_DIR / 'conformers_etkdg.sdf'
    writer = Chem.SDWriter(str(sdf_path))
    progress_path = DATA_DIR / 'conformer_progress.txt'
    progress_path.write_text('started\n', encoding='utf-8')

    index_rows = []
    fail_rows = []
    features = []

    pbar = tqdm(df.iterrows(), total=len(df), desc='Generating conformers', ncols=110)
    for i, row in pbar:
        smiles = row['smiles']
        mol, status = make_3d_mol(smiles, seed=42 + int(i))
        if mol is None:
            fail_rows.append({'row_id': int(i), 'smiles': smiles, 'status': status})
            features.append({'row_id': int(i), 'smiles': smiles, 'status': status, 'coords': None, 'coordinates': None, 'atoms': None})
            pbar.set_postfix(success=len(index_rows), failed=len(fail_rows))
            continue

        try:
            coords, atoms = extract_xyz(mol, conf_id=0)
        except Exception:
            fail_rows.append({'row_id': int(i), 'smiles': smiles, 'status': 'extract_xyz_failed'})
            features.append({'row_id': int(i), 'smiles': smiles, 'status': 'extract_xyz_failed', 'coords': None, 'coordinates': None, 'atoms': None})
            pbar.set_postfix(success=len(index_rows), failed=len(fail_rows))
            continue

        mol.SetProp('row_id', str(i))
        mol.SetProp('smiles', smiles)
        mol.SetProp('status', status)
        mol.SetProp('n_atoms', str(mol.GetNumAtoms()))
        if not pd.isna(row.get('hepg2_pIC50')):
            mol.SetProp('hepg2_pIC50', str(float(row['hepg2_pIC50'])))
        if not pd.isna(row.get('hct116_pIC50')):
            mol.SetProp('hct116_pIC50', str(float(row['hct116_pIC50'])))
        writer.write(mol)

        index_rows.append({
            'row_id': int(i),
            'smiles': smiles,
            'status': status,
            'n_atoms': int(mol.GetNumAtoms()),
            'coord_mean_abs': float(np.mean(np.abs(coords))),
            'hepg2_pIC50': row.get('hepg2_pIC50'),
            'hct116_pIC50': row.get('hct116_pIC50'),
        })
        features.append({'row_id': int(i), 'smiles': smiles, 'status': status, 'coords': coords, 'coordinates': coords, 'atoms': atoms})
        pbar.set_postfix(success=len(index_rows), failed=len(fail_rows))

        if (i + 1) % 100 == 0:
            progress_line = (
                f'processed {i + 1}/{len(df)} molecules | '
                f'success={len(index_rows)} failed={len(fail_rows)}'
            )
            print(progress_line, flush=True)
            progress_path.write_text(progress_line + '\n', encoding='utf-8')

    pbar.close()

    writer.close()

    index_df = pd.DataFrame(index_rows)
    fail_df = pd.DataFrame(fail_rows)

    out_index = DATA_DIR / 'conformer_index.csv'
    out_fail = DATA_DIR / 'conformer_failures.csv'
    out_feat = DATA_DIR / 'conformer_features.pkl'
    index_df.to_csv(out_index, index=False, encoding='utf-8-sig')
    fail_df.to_csv(out_fail, index=False, encoding='utf-8-sig')
    # 保存3D特征缓存
    import pickle
    with open(out_feat, 'wb') as f:
        pickle.dump(features, f)

    print('saved', sdf_path)
    print('saved', out_index)
    print('saved', out_fail)
    print('saved', out_feat)
    print('success', len(index_df), 'failed', len(fail_df), 'total', len(df))
    progress_path.write_text(
        f'finished success={len(index_df)} failed={len(fail_df)} total={len(df)}\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
