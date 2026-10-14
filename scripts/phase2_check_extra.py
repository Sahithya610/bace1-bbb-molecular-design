import os
from pathlib import Path
import pandas as pd

base = str(Path(__file__).resolve().parent.parent)

# Check docking_reconciliation for parsed_vina_score details
dock = pd.read_csv(os.path.join(base, 'reports/audit_corrections/docking_reconciliation.csv'))
print("=== docking_reconciliation full columns ===")
print(dock.columns.tolist())
print("\nFirst 5 rows:")
print(dock.head(5).to_string())
print("\nparsed_vina_score range:", dock['parsed_vina_score'].describe())

# Check bace1_validated columns for PAINS/Brenk/verdict
validated = pd.read_csv(os.path.join(base, 'results/docking/bace1_validated_candidates.csv'))
print("\n=== bace1_validated candidates ===")
print(validated[['canonical_smiles', 'PAINS', 'Brenk', 'Reactive', 'verdict']].to_string())

# Check oracle_exclusion
excl = pd.read_csv(os.path.join(base, 'reports/audit_corrections/oracle_exclusion_reconciliation.csv'))
print("\n=== oracle_exclusion columns ===")
print(excl.columns.tolist())
print("\nexclusion_reason value counts:")
print(excl['exclusion_reason'].value_counts())

# Check top200_scaffold for all columns
t200 = pd.read_csv(os.path.join(base, 'results/top200_scaffold_analysis.csv'))
print("\n=== top200_scaffold columns ===")
print(t200.columns.tolist())
print("\nQED column present?", 'QED' in t200.columns)
print("First row:", t200.iloc[0].to_dict())
