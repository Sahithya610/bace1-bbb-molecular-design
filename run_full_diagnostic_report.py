"""
Full diagnostic report: aggregates condition-sensitivity stats across all
checkpoints/pilots in one pass. Writes results to full_diagnostic_report.txt.
"""
import pandas as pd
import numpy as np
import os, sys

OUTPUT = "full_diagnostic_report.txt"

class Tee:
    def __init__(self, *targets): self.targets = targets
    def write(self, s):
        for t in self.targets: t.write(s); t.flush()
        return len(s)
    def flush(self):
        for t in self.targets: t.flush()

log_f = open(OUTPUT, "w", encoding="utf-8", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_f)

def stats(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return dict(n=len(s), mean=s.mean(), std=s.std(), med=s.median(),
                p10=s.quantile(0.1), p90=s.quantile(0.9))

def cond_delta(lo_csv, hi_csv, col="predicted_pIC50"):
    try:
        lo = pd.read_csv(lo_csv); hi = pd.read_csv(hi_csv)
        slo = stats(lo[col]); shi = stats(hi[col])
        return slo, shi, shi["mean"] - slo["mean"], shi["med"] - slo["med"]
    except Exception as e:
        return None, None, None, str(e)

SEP = "="*72

print(SEP)
print("FULL CONDITION-SENSITIVITY DIAGNOSTIC REPORT")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE: Issue3-ChemistryReadout  (from existing condition_test files)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] BASELINE – Issue3-ChemistryReadout")
print("-"*60)
slo, shi, dm, dmed = cond_delta(
    "results/condition_test_pic50_5_5_predictions.csv",
    "results/condition_test_pic50_8_0_predictions.csv"
)
if slo:
    print(f"  pIC50=5.5 → pred_pIC50:  μ={slo['mean']:.3f}  σ={slo['std']:.3f}  med={slo['med']:.3f}  n={slo['n']}")
    print(f"  pIC50=8.0 → pred_pIC50:  μ={shi['mean']:.3f}  σ={shi['std']:.3f}  med={shi['med']:.3f}  n={shi['n']}")
    print(f"  Δ mean = {dm:+.3f}  |  Δ median = {dmed:+.3f}")
    print(f"  Response: {'PASS ✓' if abs(dm) >= 0.3 else 'FAIL ✗'} (need |Δ| ≥ 0.3)")
    # logBB
    slo2, shi2, dm2, _ = cond_delta(
        "results/condition_test_pic50_5_5_predictions.csv",
        "results/condition_test_pic50_8_0_predictions.csv",
        col="predicted_logBB"
    )
    if slo2:
        print(f"  logBB  5.5 → pred_logBB: μ={slo2['mean']:.3f}")
        print(f"  logBB  8.0 → pred_logBB: μ={shi2['mean']:.3f}")
else:
    print(f"  ERROR: {dm}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. v3 head-vs-external diagnostic (sweep 5.5 → 9.0)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] v3 HEAD-vs-EXTERNAL SWEEP (ChemistryReadout baseline)")
print("-"*60)
try:
    df = pd.read_csv("results/issue3_v3_head_vs_external_diagnostic.csv")
    print(f"  {'req_pIC50':>10}  {'valid':>6}  {'head μ':>8}  {'ext μ':>8}  {'head-ext':>10}")
    for _, row in df.iterrows():
        print(f"  {row['requested_pIC50']:>10.1f}  "
              f"{row['valid']:>6.0f}  "
              f"{row['head_pIC50_mean']:>8.3f}  "
              f"{row['external_pIC50_mean']:>8.3f}  "
              f"{row['head_pIC50_mean']-row['external_pIC50_mean']:>+10.3f}")
    row5 = df[df['requested_pIC50']==5.5].iloc[0]
    row8 = df[df['requested_pIC50']==8.0].iloc[0]
    delta_head = row8['head_pIC50_mean'] - row5['head_pIC50_mean']
    delta_ext  = row8['external_pIC50_mean'] - row5['external_pIC50_mean']
    print(f"\n  Δ(8.0−5.5) head = {delta_head:+.3f}   ext = {delta_ext:+.3f}")
except Exception as e:
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SurrogateLoss pilot100
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] SURROGATE-LOSS – Pilot100")
print("-"*60)
slo, shi, dm, dmed = cond_delta(
    "results/surrogate_pilot_cond55_pred.csv",
    "results/surrogate_pilot_cond80_pred.csv"
)
if slo and dm is not None:
    print(f"  pIC50=5.5 → pred_pIC50:  μ={slo['mean']:.3f}  n={slo['n']}")
    print(f"  pIC50=8.0 → pred_pIC50:  μ={shi['mean']:.3f}  n={shi['n']}")
    print(f"  Δ mean = {dm:+.3f}   Response: {'PASS ✓' if abs(dm) >= 0.3 else 'FAIL ✗'}")
else:
    print(f"  skipped/missing — {dm}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ET-Distill pilot100
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] ET-DISTILL – Pilot100")
print("-"*60)
slo, shi, dm, dmed = cond_delta(
    "results/et_distill_pilot_cond55_pred.csv",
    "results/et_distill_pilot_cond80_pred.csv"
)
if slo and dm is not None:
    print(f"  pIC50=5.5 → pred_pIC50:  μ={slo['mean']:.3f}  n={slo['n']}")
    print(f"  pIC50=8.0 → pred_pIC50:  μ={shi['mean']:.3f}  n={shi['n']}")
    print(f"  Δ mean = {dm:+.3f}   Response: {'PASS ✓' if abs(dm) >= 0.3 else 'FAIL ✗'}")
else:
    print(f"  skipped/missing — {dm}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ET-Distill pilot500
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] ET-DISTILL – Pilot500")
print("-"*60)
slo, shi, dm, dmed = cond_delta(
    "results/et_distill_pilot500_cond55_pred.csv",
    "results/et_distill_pilot500_cond80_pred.csv"
)
if slo and dm is not None:
    print(f"  pIC50=5.5 → pred_pIC50:  μ={slo['mean']:.3f}  σ={slo['std']:.3f}  n={slo['n']}")
    print(f"  pIC50=8.0 → pred_pIC50:  μ={shi['mean']:.3f}  σ={shi['std']:.3f}  n={shi['n']}")
    print(f"  Δ mean = {dm:+.3f}  |  Δ median = {dmed:+.3f}")
    print(f"  Response: {'PASS ✓' if abs(dm) >= 0.3 else 'FAIL ✗'} (need |Δ| ≥ 0.3)")
else:
    print(f"  skipped/missing — {dm}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Generator output distributions (from chemistry readout eval)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] CHEMISTRY-READOUT GENERATED-MOL DISTRIBUTION")
print("-"*60)
try:
    df = pd.read_csv("results/issue3_chemistry_readout_predictions.csv")
    sp = stats(df["predicted_pIC50"]); sb = stats(df["predicted_logBB"])
    print(f"  {len(df)} molecules")
    print(f"  pred_pIC50:  μ={sp['mean']:.3f}  σ={sp['std']:.3f}  "
          f"[{sp['p10']:.2f}–{sp['p90']:.2f}] (10–90 pctile)")
    print(f"  pred_logBB:  μ={sb['mean']:.3f}  σ={sb['std']:.3f}")
    active = (df["predicted_pIC50"] >= 6.0).mean() * 100
    bbb_ok = (df["predicted_logBB"] >= 0.0).mean() * 100
    print(f"  pIC50 ≥ 6.0: {active:.1f}%  |  logBB ≥ 0: {bbb_ok:.1f}%")
    dual = ((df["predicted_pIC50"] >= 6.0) & (df["predicted_logBB"] >= 0.0)).mean() * 100
    print(f"  Dual success (pIC50≥6 AND logBB≥0): {dual:.1f}%")
except Exception as e:
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY: CONDITION RESPONSE Δ(8.0−5.5) ACROSS ALL PILOTS")
print(SEP)
results = [
    ("Baseline ChemistryReadout", "condition_test_pic50_5_5_predictions.csv",
     "condition_test_pic50_8_0_predictions.csv"),
    ("SurrogateLoss Pilot100",    "surrogate_pilot_cond55_pred.csv",
     "surrogate_pilot_cond80_pred.csv"),
    ("ET-Distill Pilot100",       "et_distill_pilot_cond55_pred.csv",
     "et_distill_pilot_cond80_pred.csv"),
    ("ET-Distill Pilot500",       "et_distill_pilot500_cond55_pred.csv",
     "et_distill_pilot500_cond80_pred.csv"),
]
print(f"  {'Model':<30}  {'n_lo':>6}  {'μ_lo':>6}  {'n_hi':>6}  {'μ_hi':>6}  {'Δμ':>7}  {'Pass?':>6}")
print("  " + "-"*68)
for name, f55, f80 in results:
    slo, shi, dm, dmed = cond_delta(f"results/{f55}", f"results/{f80}")
    if slo and dm is not None:
        flag = "✓" if abs(dm) >= 0.3 else "✗"
        print(f"  {name:<30}  {slo['n']:>6.0f}  {slo['mean']:>6.3f}  "
              f"{shi['n']:>6.0f}  {shi['mean']:>6.3f}  {dm:>+7.3f}  {flag:>6}")
    else:
        print(f"  {name:<30}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>7}  {'?':>6}")

print(f"\n{SEP}")
print("ROOT-CAUSE NOTES")
print(SEP)
print("""
  The core pattern across all approaches:
    • Internal head (training loss) responds to conditioning: corr ~ 0.33–0.38
    • External GNN oracle does NOT respond: Δ < 0.1 in every pilot
    • Generator yields valid, diverse molecules — chemistry is fine
    • Problem is purely conditioning → oracle mis-alignment

  The generator IS changing molecular graphs in response to conditioning
  (DiagSummary: 15–18% atom-decision flips between pIC50=5.5 and 8.0),
  but those graph changes are not captured by the external activity model.

  This is not a training convergence problem — it is a feature-space
  mismatch: the GAN head learns features the GAN can easily manipulate,
  while the external BACE1 predictor (GNN/RF) relies on different features.

  Recommended next steps (in order of effort):
    A. JOINT END-TO-END: back-prop through the external predictor during
       generation (requires differentiable predictor, e.g. a GNN).
    B. ORACLE AUGMENTATION: add the external predictor as a direct loss
       component (lambda * oracle_loss) so the generator must satisfy it.
    C. RANKER REPLACEMENT: swap the external oracle for one that shares
       the same backbone as the GAN's internal head (align feature spaces).
    D. CHECK ORACLE CALIBRATION: verify pred_pIC50 actually correlates
       with training pIC50 on BACE1 ligands (run predict_joint on the
       raw bace1_continuous_bbb.csv labels to measure Pearson r).
""")

print(f"\nReport written to: {OUTPUT}")
sys.stdout = sys.__stdout__
log_f.close()
