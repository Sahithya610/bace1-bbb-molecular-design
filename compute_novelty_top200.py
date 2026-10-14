import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import sys, os
from pathlib import Path

workdir = str(Path(__file__).resolve().parent)

# Load top 200
top200_path = os.path.join(workdir, 'results', 'oracle_filtered_top200_16k.csv')
df_top200 = pd.read_csv(top200_path)
print(f'Top 200 loaded: {len(df_top200)} rows')
print('Columns:', df_top200.columns.tolist())

# Identify SMILES column (try canonical_smiles, smiles, SMILES)
smiles_col = None
for c in ['canonical_smiles', 'smiles', 'SMILES']:
    if c in df_top200.columns:
        smiles_col = c
        break
if smiles_col is None:
    smiles_col = df_top200.columns[0]
    print(f'WARNING: using first column as SMILES: {smiles_col}')

# Identify pIC50 column
pic50_col = None
for c in ['predicted_pIC50', 'oracle_pIC50', 'pIC50', 'pic50']:
    if c in df_top200.columns:
        pic50_col = c
        break
if pic50_col is None:
    pic50_col = df_top200.columns[1]
    print(f'WARNING: using column as pIC50: {pic50_col}')

# Load training SMILES
train_path = os.path.join(workdir, 'results', 'bace1_activity_model', 'bace1_activity_training_table.csv')
df_train = pd.read_csv(train_path)
print(f'Training set loaded: {len(df_train)} rows')
print('Training columns:', df_train.columns.tolist())

train_smiles_col = None
for c in ['canonical_smiles', 'smiles', 'SMILES']:
    if c in df_train.columns:
        train_smiles_col = c
        break
if train_smiles_col is None:
    train_smiles_col = df_train.columns[0]
    print(f'WARNING: using training first column as SMILES: {train_smiles_col}')

# Build training fingerprints
print('Building training fingerprints...')
train_fps = []
train_smi_list = []
for smi in df_train[train_smiles_col]:
    mol = Chem.MolFromSmiles(str(smi)) if smi and str(smi) != 'nan' else None
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        train_fps.append(fp)
        train_smi_list.append(Chem.MolToSmiles(mol))
    else:
        train_fps.append(None)
        train_smi_list.append('')
train_fps_valid = [(fp, smi) for fp, smi in zip(train_fps, train_smi_list) if fp is not None]
print(f'Valid training fingerprints: {len(train_fps_valid)}')
train_fps_only = [x[0] for x in train_fps_valid]
train_smi_only = [x[1] for x in train_fps_valid]

# Analyze top 200
results = []
for idx, row in df_top200.iterrows():
    smi = str(row[smiles_col])
    pic50 = row[pic50_col] if pic50_col in df_top200.columns else None
    mol = Chem.MolFromSmiles(smi) if smi and smi != 'nan' else None
    if mol is None:
        results.append({'canonical_smiles': smi, 'predicted_pIC50': pic50,
                        'max_training_Tc': None, 'nearest_training_smiles': '',
                        'exact_training_duplicate': False, 'murcko_scaffold': ''})
        continue
    canon_smi = Chem.MolToSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps_only)
    max_tc = max(sims)
    nearest_idx = int(np.argmax(sims))
    nearest_smi = train_smi_only[nearest_idx]
    exact_dup = bool(max_tc >= 0.999)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception as e:
        scaffold = f'ERROR: {e}'
    results.append({'canonical_smiles': canon_smi, 'predicted_pIC50': pic50,
                    'max_training_Tc': max_tc, 'nearest_training_smiles': nearest_smi,
                    'exact_training_duplicate': exact_dup, 'murcko_scaffold': scaffold})

df_results = pd.DataFrame(results)
out_path = os.path.join(workdir, 'reports', 'audit_corrections', 'novelty_analysis_top200.csv')
if os.path.exists(out_path):
    print(f'NOTE: {out_path} already exists -- skipping write to avoid overwrite.')
    print('Loading existing file for summary statistics...')
    df_results_existing = pd.read_csv(out_path)
    # Map column names if needed
    tc_col = 'max_training_Tc' if 'max_training_Tc' in df_results_existing.columns else None
    dup_col = 'exact_training_duplicate' if 'exact_training_duplicate' in df_results_existing.columns else None
    scaffold_col = 'murcko_scaffold' if 'murcko_scaffold' in df_results_existing.columns else None
    n_exact_dup = df_results_existing[dup_col].sum() if dup_col else df_results['exact_training_duplicate'].sum()
    unique_scaffolds = df_results_existing[scaffold_col].nunique() if scaffold_col else df_results['murcko_scaffold'].nunique()
    mean_max_tc = df_results_existing[tc_col].mean() if tc_col else df_results['max_training_Tc'].mean()
else:
    df_results.to_csv(out_path, index=False)
    print(f'Saved novelty analysis to {out_path}')
    n_exact_dup = df_results['exact_training_duplicate'].sum()
    unique_scaffolds = df_results['murcko_scaffold'].nunique()
    mean_max_tc = df_results['max_training_Tc'].mean()

# Summary statistics
print('Computing mean pairwise Tanimoto among top 200...')
top200_fps = []
for smi in df_results['canonical_smiles']:
    mol = Chem.MolFromSmiles(str(smi)) if smi and str(smi) != 'nan' else None
    if mol:
        top200_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    else:
        top200_fps.append(None)
top200_fps_valid = [fp for fp in top200_fps if fp is not None]
pairwise_tcs = []
for i in range(len(top200_fps_valid)):
    sims = DataStructs.BulkTanimotoSimilarity(top200_fps_valid[i], top200_fps_valid[i+1:])
    pairwise_tcs.extend(sims)
mean_pairwise_tc = np.mean(pairwise_tcs) if pairwise_tcs else 0.0

print(f'\n=== NOVELTY SUMMARY ===')
print(f'Total analyzed: {len(df_results)}')
print(f'Exact training duplicates (Tc>=0.999): {n_exact_dup}')
print(f'Unique Murcko scaffolds: {unique_scaffolds}')
print(f'Mean max training Tc: {mean_max_tc:.4f}')
print(f'Mean pairwise Tc (top200): {mean_pairwise_tc:.4f}')
print(f'Claimed: 186/200 unique scaffolds, mean pairwise Tc=0.146')
print(f'Verified: {unique_scaffolds}/200 unique scaffolds, mean pairwise Tc={mean_pairwise_tc:.3f}')
