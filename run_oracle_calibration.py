"""
Oracle calibration check: run the external predict_joint model on the
known BACE1 training molecules and measure how well its predicted_pIC50
correlates with the ground-truth pIC50 labels in bace1_continuous_bbb.csv.

Also checks pIC50 range and distribution of the training set.
"""
import subprocess, sys, os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

OUTPUT = "oracle_calibration_report.txt"
class Tee:
    def __init__(self, *targets): self.targets = targets
    def write(self, s):
        for t in self.targets: t.write(s); t.flush()
        return len(s)
    def flush(self):
        for t in self.targets: t.flush()

log_f = open(OUTPUT, "w", encoding="utf-8", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_f)
SEP = "="*72

print(SEP)
print("ORACLE CALIBRATION REPORT")
print(SEP)

# ── Step 1: describe the ground-truth label distribution ──────────────────
labels = pd.read_csv("druggen_data/bace1_continuous_bbb.csv")
pic50  = labels["pIC50"].dropna()
logbb  = labels["predicted_logBB"].dropna()

print(f"\n[A] BACE1 ground-truth pIC50 distribution (n={len(pic50)})")
print(f"    μ={pic50.mean():.3f}  σ={pic50.std():.3f}  "
      f"min={pic50.min():.2f}  max={pic50.max():.2f}")
print(f"    p10={pic50.quantile(0.10):.2f}  p25={pic50.quantile(0.25):.2f}  "
      f"med={pic50.median():.2f}  p75={pic50.quantile(0.75):.2f}  "
      f"p90={pic50.quantile(0.90):.2f}")
print(f"    Fraction ≥6.0: {(pic50>=6.0).mean()*100:.1f}%  "
      f"≥7.0: {(pic50>=7.0).mean()*100:.1f}%  "
      f"≥8.0: {(pic50>=8.0).mean()*100:.1f}%")

# ── Step 2: run predict_joint on the training molecules ───────────────────
print(f"\n[B] Running predict_joint on bace1_continuous_bbb.csv …")
TMP_INPUT  = "results/_oracle_cal_input.csv"
TMP_OUTPUT = "results/_oracle_cal_pred.csv"

# predict_joint expects columns: smiles, target, requested_pIC50, requested_logBB
cal_df = labels[["canonical_smiles","pIC50","predicted_logBB"]].copy()
cal_df = cal_df.rename(columns={"canonical_smiles":"smiles"})
cal_df["target"]        = "BACE1"
cal_df["requested_pIC50"]  = cal_df["pIC50"]       # use GT as requested
cal_df["requested_logBB"]  = cal_df["predicted_logBB"]
cal_df.to_csv(TMP_INPUT, index=False)

py = sys.executable
result = subprocess.run(
    [py, "predict_joint.py", "--input", TMP_INPUT, "--output", TMP_OUTPUT],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"    ERROR running predict_joint:\n{result.stderr}")
    sys.exit(1)
print("    Done.")

# ── Step 3: compute correlation ───────────────────────────────────────────
pred = pd.read_csv(TMP_OUTPUT)
# merge on smiles
merged = pred.merge(
    labels[["canonical_smiles","pIC50"]],
    on="canonical_smiles", how="inner"
)
print(f"\n[C] Oracle prediction quality (n={len(merged)} matched)")

if len(merged) < 10:
    print("    WARNING: very few matches — SMILES canonicalization mismatch?")
    # try merging on raw SMILES
    merged2 = pred.merge(
        labels[["canonical_smiles","pIC50"]],
        left_on="smiles", right_on="canonical_smiles", how="inner"
    )
    print(f"    Fallback raw-smiles merge: {len(merged2)} matches")
    if len(merged2) > len(merged):
        merged = merged2

if len(merged) >= 5:
    gt  = merged["pIC50"].values
    pr  = merged["predicted_pIC50"].values
    r,  p_r  = pearsonr(gt, pr)
    rho, p_s = spearmanr(gt, pr)
    rmse = np.sqrt(np.mean((gt - pr)**2))
    mae  = np.mean(np.abs(gt - pr))
    bias = np.mean(pr - gt)

    print(f"    Pearson  r  = {r:.4f}  (p={p_r:.2e})")
    print(f"    Spearman ρ  = {rho:.4f}  (p={p_s:.2e})")
    print(f"    RMSE        = {rmse:.3f}")
    print(f"    MAE         = {mae:.3f}")
    print(f"    Bias(pr-gt) = {bias:+.3f}  (systematic over/under)")

    print(f"\n    pred_pIC50 distribution over training set:")
    pr_s = pd.Series(pr)
    print(f"    μ={pr_s.mean():.3f}  σ={pr_s.std():.3f}  "
          f"min={pr_s.min():.2f}  max={pr_s.max():.2f}")
    print(f"    Fraction ≥6.0: {(pr_s>=6.0).mean()*100:.1f}%  "
          f"≥7.0: {(pr_s>=7.0).mean()*100:.1f}%")

    print(f"\n[D] CALIBRATION VERDICT")
    print("-"*60)
    if r >= 0.7:
        print(f"    ✓ Oracle is well-calibrated (r={r:.3f} ≥ 0.70)")
        print(f"    The condition-response failure is NOT due to a bad oracle.")
        print(f"    Root cause: feature-space mismatch between GAN head and oracle.")
    elif r >= 0.4:
        print(f"    ⚠ Oracle is moderately correlated (r={r:.3f})")
        print(f"    Partial calibration — oracle may not reliably rank molecules.")
    else:
        print(f"    ✗ Oracle is POORLY calibrated (r={r:.3f} < 0.40)")
        print(f"    The external predictor is unreliable on generated molecules.")
        print(f"    The Δ=0.07 condition response may be noise, not a training failure.")
else:
    print("    Not enough matched rows to compute correlation.")

# cleanup
for f in [TMP_INPUT]: os.remove(f)

print(f"\n{SEP}")
print(f"Report written to: {OUTPUT}")
sys.stdout = sys.__stdout__
log_f.close()
