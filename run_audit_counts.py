"""Read-only audit: verify all counts from actual saved files."""
import pandas as pd
import numpy as np

# ── 1. 16384-molecule pool ─────────────────────────────────────────
pool = pd.read_csv("results/oracle_pool_16384.csv")
pred = pd.read_csv("results/oracle_pool_16384_pred.csv")

n_total      = len(pool)
n_valid      = pool["valid"].sum()
n_oracle     = len(pred)
n_unique     = pred["canonical_smiles"].nunique()
n_ge60       = (pred["predicted_pIC50"] >= 6.0).sum()
n_ge70       = (pred["predicted_pIC50"] >= 7.0).sum()
n_ge76576    = (pred["predicted_pIC50"] >= 7.6576).sum()
max_pic50    = pred["predicted_pIC50"].max()
max_smi      = pred.loc[pred["predicted_pIC50"].idxmax(), "canonical_smiles"]

print("=== 16384-POOL GENERATION COUNTS (from oracle_pool_16384*.csv) ===")
print(f"  Total rows in generation CSV:     {n_total}")
print(f"  Rows with valid=True:             {int(n_valid)}")
print(f"  Rows in prediction CSV:           {n_oracle}")
print(f"  Unique canonical SMILES:          {n_unique}")
print(f"  Predicted pIC50 >= 6.0:           {n_ge60}")
print(f"  Predicted pIC50 >= 7.0:           {n_ge70}")
print(f"  Predicted pIC50 >= 7.6576:        {n_ge76576}  (original joint threshold)")
print(f"  Max predicted pIC50:              {max_pic50:.4f}")
print(f"  SMILES of max:                    {max_smi}")

# BBB on max
max_logBB = pred.loc[pred["predicted_pIC50"].idxmax(), "predicted_logBB"]
print(f"  predicted_logBB of max:           {max_logBB:.4f}  (BBB+ requires >= 0)")
print(f"  Max molecule meets BBB criterion? {'YES' if max_logBB >= 0 else 'NO'}")

# ── 2. Molecules meeting BOTH original thresholds ─────────────────
both = pred[(pred["predicted_pIC50"] >= 7.6576) & (pred["predicted_logBB"] >= 0.0)]
print(f"\n  Molecules meeting BOTH thresholds (pIC50>=7.6576 AND logBB>=0): {len(both)}")

# ── 3. Validated candidates ────────────────────────────────────────
val = pd.read_csv("results/docking/bace1_validated_candidates.csv")
print(f"\n=== VALIDATED DUAL-SUCCESS CANDIDATES ===")
print(f"  Rows in validated CSV:            {len(val)}")
print(f"  CLEAN (no alerts):                {(val['verdict'].str.startswith('CLEAN')).sum()}")
print(f"  CLEAN, QED>=0.5:                  {((val['verdict']=='CLEAN') & (val['QED']>=0.5)).sum()}")
print(f"  Any meeting pIC50>=7.6576:        {(val['oracle_pIC50']>=7.6576).sum()}")
print(f"  pIC50 range in validated set:     {val['oracle_pIC50'].min():.3f} – {val['oracle_pIC50'].max():.3f}")

# ── 4. Docking results ─────────────────────────────────────────────
dock = pd.read_csv("results/docking/bace1_final_ranked.csv")
print(f"\n=== DOCKING RESULTS ===")
print(f"  Total rows:                       {len(dock)}")
print(f"  Successfully docked (vina notna): {dock['vina_score'].notna().sum()}")
print(f"  BBB_positive (logBB>=0):          {dock['BBB_positive'].sum()}")
vina_bbb = dock[(dock['vina_score'] <= -7.5) & (dock['BBB_positive'])]
print(f"  Vina<=-7.5 AND BBB+:              {len(vina_bbb)}")
print(f"  Best Vina score:                  {dock['vina_score'].min():.3f}")
print(f"  Mean Vina score:                  {dock['vina_score'].mean():.3f}")

# ── 5. Original threshold check on docked set ─────────────────────
meets_both_docked = dock[(dock['oracle_pIC50']>=7.6576) & (dock['BBB_positive'])]
print(f"  Docked mols meeting pIC50>=7.6576 AND BBB+: {len(meets_both_docked)}")

# ── 6. Oracle test-set performance ────────────────────────────────
import json
with open("results/bace1_activity_model/model_metadata.json") as f:
    meta = json.load(f)
print(f"\n=== ORACLE MODEL METADATA ===")
print(f"  Model type:       {meta['selected_model']}")
print(f"  Fingerprint:      {meta['fingerprint']}")
print(f"  Split strategy:   scaffold-based (from metadata keys)")
print(f"  Train/val/test:   {meta['train_samples']}/{meta['validation_samples']}/{meta['test_samples']}")
print(f"  Train scaffolds:  {meta['train_scaffolds']}")
print(f"  Test scaffolds:   {meta['test_scaffolds']} (held-out scaffolds)")
print(f"  Test R²:          {meta['test_results']['R2']:.4f}")
print(f"  Test RMSE:        {meta['test_results']['RMSE']:.4f}")
print(f"  Test MAE:         {meta['test_results']['MAE']:.4f}")
print(f"  IMPORTANT: r=0.901 was reported from training-set self-prediction.")
print(f"  Independent scaffold-split test R² = {meta['test_results']['R2']:.4f} (sqrt = {meta['test_results']['R2']**0.5:.3f})")
print(f"  This is the honest calibration figure for novel generated molecules.")
