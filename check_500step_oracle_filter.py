"""
Understand why predict_joint only passed n=2 molecules for the 500-step
cond=5.5 result. Check the raw generated CSV and run predict_joint with
verbose output to see what's being filtered.
"""
import pandas as pd
import subprocess, sys

# Check the 500-step generation
gen = pd.read_csv("results/vg2_c5_p5_cond55.csv")
print(f"500-step cond=5.5 generation: {len(gen)} rows")
valid = gen[gen["valid"] == True]
print(f"  Valid molecules: {len(valid)}")
smiles_col = valid["smiles"].dropna()
smiles_col = smiles_col[smiles_col != ""]
print(f"  Non-empty SMILES: {len(smiles_col)}")
tiny = smiles_col[smiles_col.str.len() <= 3]
real = smiles_col[smiles_col.str.len() > 3]
print(f"  Tiny (len≤3): {len(tiny)}")
print(f"  Real chemistry (len>3): {len(real)}")
if len(real) > 0:
    print(f"  Sample real SMILES (first 5):")
    for s in real.head(5):
        print(f"    {s}")

# Check the prediction output
pred = pd.read_csv("results/vg2_c5_p5_cond55_pred.csv")
print(f"\n500-step cond=5.5 predictions: {len(pred)} rows")
print(f"  Columns: {list(pred.columns)}")
print(pred.head(5).to_string())
