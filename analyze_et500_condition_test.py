"""
Analyze condition sensitivity for ET-distill 500-step pilot.
"""
import pandas as pd
import numpy as np

print("="*70)
print("ET-DISTILL 500-STEP PILOT: CONDITION SENSITIVITY ANALYSIS")
print("="*70)

# Load prediction files
cond55 = pd.read_csv("results/et_distill_pilot500_cond55_pred.csv")
cond80 = pd.read_csv("results/et_distill_pilot500_cond80_pred.csv")

print(f"\nGenerated molecules:")
print(f"  pIC50=5.5 requested: {len(cond55):4d} molecules")
print(f"  pIC50=8.0 requested: {len(cond80):4d} molecules")

# Stats for pIC50=5.5
print(f"\n{'Condition: pIC50=5.5'}")
print(f"  predicted_pIC50: μ={cond55['predicted_pIC50'].mean():.3f}, "
      f"σ={cond55['predicted_pIC50'].std():.3f}, "
      f"med={cond55['predicted_pIC50'].median():.3f}")
print(f"  predicted_logBB: μ={cond55['predicted_logBB'].mean():.3f}, "
      f"σ={cond55['predicted_logBB'].std():.3f}, "
      f"med={cond55['predicted_logBB'].median():.3f}")

# Stats for pIC50=8.0
print(f"\n{'Condition: pIC50=8.0'}")
print(f"  predicted_pIC50: μ={cond80['predicted_pIC50'].mean():.3f}, "
      f"σ={cond80['predicted_pIC50'].std():.3f}, "
      f"med={cond80['predicted_pIC50'].median():.3f}")
print(f"  predicted_logBB: μ={cond80['predicted_logBB'].mean():.3f}, "
      f"σ={cond80['predicted_logBB'].std():.3f}, "
      f"med={cond80['predicted_logBB'].median():.3f}")

# Compare
delta_pic50_mean = cond80['predicted_pIC50'].mean() - cond55['predicted_pIC50'].mean()
delta_pic50_median = cond80['predicted_pIC50'].median() - cond55['predicted_pIC50'].median()

print(f"\n{'Δ (8.0 - 5.5):'}")
print(f"  Mean pIC50:   {delta_pic50_mean:+.3f}")
print(f"  Median pIC50: {delta_pic50_median:+.3f}")

# Decision criteria
print(f"\n{'='*70}")
print(f"DECISION CRITERIA:")
print(f"{'='*70}")

# 1. Validity (should be ~100% given successful generation)
valid_55 = cond55['valid'].sum()
valid_80 = cond80['valid'].sum()
print(f"1. Validity: {valid_55}/{len(cond55)} ({100*valid_55/len(cond55):.1f}%) @ 5.5, "
      f"{valid_80}/{len(cond80)} ({100*valid_80/len(cond80):.1f}%) @ 8.0")

# 2. Condition response magnitude
response_ok = abs(delta_pic50_mean) > 0.3
print(f"2. Condition response: Δμ={delta_pic50_mean:+.3f} "
      f"{'✓ PASS' if response_ok else '✗ FAIL'} (need |Δ| > 0.3)")

# 3. Direction check
direction_ok = delta_pic50_mean > 0  # Should increase from 5.5 to 8.0
print(f"3. Response direction: {'✓ PASS (positive)' if direction_ok else '✗ FAIL (wrong sign)'}")

# Overall decision
all_pass = response_ok and direction_ok and (valid_55/len(cond55) > 0.95) and (valid_80/len(cond80) > 0.95)

print(f"\n{'='*70}")
if all_pass:
    print("DECISION: ✓ PROCEED TO FULL TRAINING")
    print("The 500-step pilot shows sufficient condition response.")
    print("Run: python run_et_distill_full.py")
else:
    print("DECISION: ✗ DO NOT PROCEED")
    print("The 500-step pilot did not meet decision criteria.")
    print("Investigate before committing to full 10-epoch training.")
print(f"{'='*70}")
