"""
Analyse oracle calibration from already-generated prediction file.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

SEP = "="*72
OUTPUT = "oracle_calibration_report.txt"

labels = pd.read_csv("druggen_data/bace1_continuous_bbb.csv")
pred   = pd.read_csv("results/_oracle_cal_pred.csv")

print(f"labels columns: {list(labels.columns)}")
print(f"pred   columns: {list(pred.columns)}")
print(f"labels rows: {len(labels)}   pred rows: {len(pred)}")

# merge on canonical_smiles
merged = pred.merge(labels[["canonical_smiles","pIC50"]].rename(columns={"pIC50":"gt_pIC50"}),
                    on="canonical_smiles", how="inner")
print(f"merged rows (canonical_smiles join): {len(merged)}")

if len(merged) == 0:
    merged = pred.merge(labels[["canonical_smiles","pIC50"]].rename(columns={"pIC50":"gt_pIC50"}),
                        left_on="smiles", right_on="canonical_smiles", how="inner")
    print(f"merged rows (smiles fallback): {len(merged)}")

gt = merged["gt_pIC50"].values
pr = merged["predicted_pIC50"].values

r,   p_r  = pearsonr(gt, pr)
rho, p_s  = spearmanr(gt, pr)
rmse = np.sqrt(np.mean((gt - pr)**2))
mae  = np.mean(np.abs(gt - pr))
bias = np.mean(pr - gt)

print(f"\n{SEP}")
print("ORACLE CALIBRATION – BACE1 training set (n={})".format(len(merged)))
print(SEP)
print(f"  Pearson  r  = {r:.4f}  (p={p_r:.2e})")
print(f"  Spearman ρ  = {rho:.4f}  (p={p_s:.2e})")
print(f"  RMSE        = {rmse:.3f}")
print(f"  MAE         = {mae:.3f}")
print(f"  Bias(pr-gt) = {bias:+.3f}")

pr_s = pd.Series(pr)
print(f"\n  pred_pIC50 distribution:")
print(f"  μ={pr_s.mean():.3f}  σ={pr_s.std():.3f}  "
      f"min={pr_s.min():.2f}  max={pr_s.max():.2f}")
print(f"  Fraction ≥6.0: {(pr_s>=6.0).mean()*100:.1f}%  "
      f"≥7.0: {(pr_s>=7.0).mean()*100:.1f}%")

print(f"\n  ground-truth pIC50 distribution:")
gt_s = pd.Series(gt)
print(f"  μ={gt_s.mean():.3f}  σ={gt_s.std():.3f}  "
      f"min={gt_s.min():.2f}  max={gt_s.max():.2f}")

print(f"\n{SEP}")
print("VERDICT")
print(SEP)
if r >= 0.70:
    print(f"  ✓ Oracle is well-calibrated (r={r:.3f})")
    print(f"  The Δ≈0.07 condition response is a real training failure.")
    print(f"  Root cause: GAN head learns proxy features, oracle requires")
    print(f"  pharmacophoric features the GAN is not optimising for.")
elif r >= 0.40:
    print(f"  ⚠ Oracle moderately calibrated (r={r:.3f})")
    print(f"  Partial reliability — weak signal in generated molecules.")
else:
    print(f"  ✗ Oracle is POORLY calibrated (r={r:.3f})")
    print(f"  The oracle cannot rank molecules reliably.")
    print(f"  The Δ≈0.07 result may be noise, not a conditioning failure.")
    print(f"  Priority: replace or retrain the oracle before further work.")

# note predicted range vs generated mol range
gen_lo = pd.read_csv("results/et_distill_pilot500_cond55_pred.csv")["predicted_pIC50"]
gen_hi = pd.read_csv("results/et_distill_pilot500_cond80_pred.csv")["predicted_pIC50"]
print(f"\n  Generated mol oracle range:")
print(f"  cond=5.5 → pred μ={gen_lo.mean():.3f}  range [{gen_lo.min():.2f}–{gen_lo.max():.2f}]")
print(f"  cond=8.0 → pred μ={gen_hi.mean():.3f}  range [{gen_hi.min():.2f}–{gen_hi.max():.2f}]")
print(f"\n  Training set oracle range: [{pr_s.min():.2f}–{pr_s.max():.2f}]")
print(f"  ▶ Generated mols sit at μ≈5.45 vs training set μ≈{pr_s.mean():.2f}")
print(f"    (oracle range contraction = oracle collapses range of generated mols?)")
