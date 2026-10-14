from pathlib import Path
import sys
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
csv_path = ROOT / 'results' / 'baseline_screen_50000_generated.csv'
out_path = ROOT / 'analyze_generation_output.txt'

lines = []

df = pd.read_csv(csv_path)

total_rows = len(df)
valid_df = df[df['valid'] == True]
valid_count = len(valid_df)
validity_fraction = valid_count / total_rows if total_rows else 0.0

lines.append(f"Total rows: {total_rows}")
lines.append(f"Valid rows (valid==True): {valid_count}")
lines.append(f"Validity fraction: {validity_fraction:.4%}")

# First 3 SMILES
lines.append("\nFirst 3 SMILES:")
smiles_sample = df['smiles'].dropna().head(3)
for i, s in enumerate(smiles_sample, 1):
    lines.append(f"  {i}: {s}")

# Atom-class collapse check: valid rows where SMILES has only 1 unique heavy atom symbol
collapse_count = 0
non_collapse_count = 0
error_count = 0
for smi in valid_df['smiles'].dropna():
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            error_count += 1
            continue
        symbols = set(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0)
        if len(symbols) <= 1:
            collapse_count += 1
        else:
            non_collapse_count += 1
    except Exception:
        error_count += 1

lines.append(f"\nAtom-class collapse check (valid molecules with only 1 unique heavy atom symbol):")
lines.append(f"  Single-element molecules: {collapse_count}")
lines.append(f"  Multi-element molecules:  {non_collapse_count}")
lines.append(f"  Parse errors:             {error_count}")
collapse_fraction = collapse_count / valid_count if valid_count else 0.0
lines.append(f"  Collapse fraction: {collapse_fraction:.4%}")

if collapse_fraction > 0.50:
    lines.append("\nATOM CLASS COLLAPSE DETECTED: >50% of valid molecules are single-element.")
    exit_code = 2
else:
    lines.append("\nNo atom-class collapse detected.")
    exit_code = 0

with open(out_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
    f.write(f"ExitCode:{exit_code}\n")

sys.exit(exit_code)
