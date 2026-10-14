"""Quick check: what normalized values does the clamp produce for key pIC50 targets?"""
PIC50_MEAN, PIC50_STD = 6.691315, 1.308678
BBB_MEAN,   BBB_STD   = -0.077873, 0.751623

def norm_pic50(v):
    raw = (v - PIC50_MEAN) / PIC50_STD
    return max(-1.5, min(1.5, raw)), raw

print(f"{'pIC50':>8}  {'raw_norm':>10}  {'clamped':>10}  {'clipped?':>9}")
print("-" * 45)
for v in [4.0, 5.0, 5.5, 6.0, 6.5, 6.691, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 12.0]:
    clamped, raw = norm_pic50(v)
    clip = "YES" if clamped != raw else ""
    print(f"{v:>8.1f}  {raw:>+10.4f}  {clamped:>+10.4f}  {clip:>9}")

print()
print("Training data p2.5 / p97.5 normalization range:")
import pandas as pd
labels = pd.read_csv("druggen_data/bace1_continuous_bbb.csv")
normed = (labels["pIC50"] - PIC50_MEAN) / PIC50_STD
print(f"  p2.5  = {normed.quantile(0.025):+.3f}")
print(f"  p97.5 = {normed.quantile(0.975):+.3f}")
print(f"  min   = {normed.min():+.3f}")
print(f"  max   = {normed.max():+.3f}")
print(f"  Fraction clipped by ±1.5: {((normed < -1.5) | (normed > 1.5)).mean()*100:.1f}%")
