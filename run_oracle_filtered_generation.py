"""
Post-hoc oracle-filtered generation.

Strategy:
  1. Generate a large pool (8192 molecules) from the baseline checkpoint
     at a neutral condition (pIC50=7.0, logBB=0.0)
  2. Run the calibrated oracle (r=0.901) on every valid molecule
  3. Filter and rank by predicted pIC50
  4. Report top-N and save to results/oracle_filtered_top.csv

No conditioning tricks, no proxy heads, no REINFORCE.
The oracle directly selects the best molecules from the pool.
"""
import os, sys, subprocess
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT = "experiments/models/Issue3-ChemistryReadout/joint_generator.pt"
POOL_FILE  = "results/oracle_pool_16384.csv"
PRED_FILE  = "results/oracle_pool_16384_pred.csv"
TOP_FILE   = "results/oracle_filtered_top200_16k.csv"
TOP_N      = 200

py = sys.executable

# ── Step 1: Generate 8192 molecules ───────────────────────────────────
print("="*60)
print("STEP 1: Generating 16384 molecules from baseline checkpoint")
print("="*60)
r = subprocess.run([
    py, "generate_joint_molecules.py",
    "--checkpoint", CHECKPOINT,
    "--desired_pIC50", "7.0",   # neutral mid-range condition
    "--desired_logBB", "0.0",
    "--batch_size", "128",
    "--num_batches", "128",      # 128 * 128 = 16384
    "--seed", "42",
    "--output", POOL_FILE,
], cwd=os.path.dirname(os.path.abspath(__file__)))

if r.returncode != 0:
    print("Generation failed. Exiting.")
    sys.exit(1)

gen = pd.read_csv(POOL_FILE)
n_valid = gen["valid"].sum()
print(f"Generated: {len(gen)} total, {n_valid} valid")

# ── Step 2: Oracle predictions ────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Running oracle on all valid molecules")
print("="*60)
subprocess.run([
    py, "predict_joint.py",
    "--input",  POOL_FILE,
    "--output", PRED_FILE,
], cwd=os.path.dirname(os.path.abspath(__file__)))

# ── Step 3: Filter and rank ───────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Filtering and ranking")
print("="*60)
pred = pd.read_csv(PRED_FILE)

print(f"Oracle scored: {len(pred)} molecules")
print(f"pred_pIC50: μ={pred['predicted_pIC50'].mean():.3f}  "
      f"σ={pred['predicted_pIC50'].std():.3f}  "
      f"max={pred['predicted_pIC50'].max():.3f}")
print(f"pred_logBB: μ={pred['predicted_logBB'].mean():.3f}")

# Rank by pIC50 descending
pred_sorted = pred.sort_values("predicted_pIC50", ascending=False).reset_index(drop=True)

# Distribution across thresholds
for thresh in [6.0, 6.5, 7.0, 7.5, 8.0]:
    n = (pred["predicted_pIC50"] >= thresh).sum()
    pct = 100 * n / len(pred)
    print(f"  pIC50 ≥ {thresh}: {n:4d} ({pct:.1f}%)")

# Top-N
top = pred_sorted.head(TOP_N).copy()
top.to_csv(TOP_FILE, index=False)

# ── Step 4: Report ────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"TOP {TOP_N} MOLECULES")
print("="*60)
print(f"pIC50 range: [{top['predicted_pIC50'].min():.3f}, "
      f"{top['predicted_pIC50'].max():.3f}]")
print(f"pIC50 mean:  {top['predicted_pIC50'].mean():.3f}")
print(f"logBB mean:  {top['predicted_logBB'].mean():.3f}")
print(f"logBB ≥ 0 (BBB+): {(top['predicted_logBB'] >= 0).sum()}/{TOP_N} "
      f"({100*(top['predicted_logBB']>=0).mean():.0f}%)")
dual = ((top['predicted_pIC50'] >= 7.0) & (top['predicted_logBB'] >= 0)).sum()
print(f"Dual success (pIC50≥7.0 AND logBB≥0): {dual}/{TOP_N}")

print(f"\nTop 10 molecules:")
print(top[["canonical_smiles", "predicted_pIC50", "predicted_logBB"]].head(10).to_string(index=False))

print(f"\nSaved top {TOP_N} to: {TOP_FILE}")
