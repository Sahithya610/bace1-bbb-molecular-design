import pandas as pd
import numpy as np

alignment_file = "results/bace1_final_affinity_alignment.csv"

df = pd.read_csv(alignment_file)

if "pIC50" not in df.columns:
    raise ValueError("pIC50 column not found.")

if df["pIC50"].isna().any():
    raise ValueError("Missing pIC50 values found.")

p = df["pIC50"].astype(float)

print("=" * 70)
print("BACE1 AFFINITY CONDITION STATISTICS")
print("=" * 70)

print(f"Graphs represented: {len(df)}")
print(f"Mean pIC50       : {p.mean():.6f}")
print(f"Std pIC50        : {p.std():.6f}")
print(f"Median pIC50     : {p.median():.6f}")
print(f"Minimum pIC50    : {p.min():.6f}")
print(f"25th percentile  : {p.quantile(0.25):.6f}")
print(f"75th percentile  : {p.quantile(0.75):.6f}")
print(f"90th percentile  : {p.quantile(0.90):.6f}")
print(f"Maximum pIC50    : {p.max():.6f}")

print()
print("PROPOSED GENERATION CONDITIONS")
print("-" * 70)

mean = p.mean()
std = p.std()

for desired in [p.median(), p.quantile(0.75), 8.0, p.quantile(0.90)]:
    normalized = (desired - mean) / std
    print(
        f"pIC50={desired:.4f} -> normalized={normalized:.4f}"
    )

print()
print("=" * 70)