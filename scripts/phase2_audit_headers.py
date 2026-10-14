import os
from pathlib import Path
import pandas as pd

base = str(Path(__file__).resolve().parent.parent)

files = {
    'oracle_pool_16384_pred': 'results/oracle_pool_16384_pred.csv',
    'oracle_filtered_top200': 'results/oracle_filtered_top200_16k.csv',
    'top200_scaffold': 'results/top200_scaffold_analysis.csv',
    'docking_recon': 'reports/audit_corrections/docking_reconciliation.csv',
    'oracle_exclusion': 'reports/audit_corrections/oracle_exclusion_reconciliation.csv',
    'bace1_validated': 'results/docking/bace1_validated_candidates.csv',
    'bace1_training': 'results/bace1_activity_model/bace1_activity_training_table.csv',
}

for name, path in files.items():
    full = os.path.join(base, path)
    if not os.path.exists(full):
        print(f'=== {name} === MISSING: {full}')
        continue
    df = pd.read_csv(full, nrows=3)
    print(f'=== {name} ===')
    print(f'  Columns ({len(df.columns)}): {list(df.columns)}')
    # print first row values
    if len(df) > 0:
        for col in df.columns[:6]:
            print(f'    {col}: {df[col].iloc[0]}')
    print()

# Full shape for primary files
for name, path in files.items():
    full = os.path.join(base, path)
    if os.path.exists(full):
        df = pd.read_csv(full)
        print(f'{name}: {df.shape}')
