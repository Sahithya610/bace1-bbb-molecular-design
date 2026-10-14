import pandas as pd

f = "results/baseline_screen_50000_predictions.csv"
df = pd.read_csv(f)

print(f"Total rows: {len(df)}")
print(f"Columns present: {list(df.columns)}")
has_pic50 = "predicted_pIC50" in df.columns
has_logbb = "predicted_logBB" in df.columns
has_smiles = "canonical_smiles" in df.columns
print(f"predicted_pIC50 column: {has_pic50}")
print(f"predicted_logBB column: {has_logbb}")
print(f"canonical_smiles column: {has_smiles}")

if not has_pic50:
    print("ERROR: predicted_pIC50 column missing")
    raise SystemExit(1)

print()
p = df["predicted_pIC50"]
print(f"predicted_pIC50  min={p.min():.4f}  max={p.max():.4f}  mean={p.mean():.4f}  median={p.median():.4f}  std={p.std():.4f}")
b = df["predicted_logBB"]
print(f"predicted_logBB  min={b.min():.4f}  max={b.max():.4f}  mean={b.mean():.4f}  median={b.median():.4f}  std={b.std():.4f}")

print()
n = len(df)
n6   = (p >= 6.0).sum()
n7   = (p >= 7.0).sum()
n765 = (p >= 7.6576).sum()
nbbb = (b >= 0.0).sum()
njoint = ((p >= 7.6576) & (b >= 0.0)).sum()

print(f"pIC50 >= 6.0:    {n6:6d}  ({100*n6/n:.3f}%)")
print(f"pIC50 >= 7.0:    {n7:6d}  ({100*n7/n:.3f}%)")
print(f"pIC50 >= 7.6576: {n765:6d}  ({100*n765/n:.3f}%)")
print(f"logBB >= 0.0:    {nbbb:6d}  ({100*nbbb/n:.3f}%)")
print(f"JOINT (both):    {njoint:6d}  ({100*njoint/n:.3f}%)")

if njoint > 0:
    top = df[(p >= 7.6576) & (b >= 0.0)].sort_values("predicted_pIC50", ascending=False).head(10)
    print()
    print("Top joint-pass molecules:")
    print(top[["canonical_smiles", "predicted_pIC50", "predicted_logBB"]].to_string(index=False))
