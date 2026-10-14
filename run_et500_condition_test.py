"""
Run condition sensitivity test for ET-distill 500-step pilot.
Generates molecules at two different pIC50 conditions and predicts their properties.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run_cmd(cmd):
    """Run command and stream output."""
    print(f"\n{'='*70}\nRunning: {' '.join(cmd)}\n{'='*70}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        sys.exit(1)
    return result

# Python interpreter
py = sys.executable

# Generation at pIC50=5.5
run_cmd([
    py, "generate_joint_molecules.py",
    "--checkpoint", "experiments/models/Issue3-ETDistill-Pilot500/joint_generator.pt",
    "--desired_pIC50", "5.5",
    "--desired_logBB", "0.0",
    "--batch_size", "128",
    "--num_batches", "8",
    "--seed", "42",
    "--output", "results/et_distill_pilot500_cond55.csv"
])

# Generation at pIC50=8.0
run_cmd([
    py, "generate_joint_molecules.py",
    "--checkpoint", "experiments/models/Issue3-ETDistill-Pilot500/joint_generator.pt",
    "--desired_pIC50", "8.0",
    "--desired_logBB", "0.0",
    "--batch_size", "128",
    "--num_batches", "8",
    "--seed", "42",
    "--output", "results/et_distill_pilot500_cond80.csv"
])

# Prediction for pIC50=5.5
run_cmd([
    py, "predict_joint.py",
    "--input", "results/et_distill_pilot500_cond55.csv",
    "--output", "results/et_distill_pilot500_cond55_pred.csv"
])

# Prediction for pIC50=8.0
run_cmd([
    py, "predict_joint.py",
    "--input", "results/et_distill_pilot500_cond80.csv",
    "--output", "results/et_distill_pilot500_cond80_pred.csv"
])

print("\n" + "="*70)
print("CONDITION TEST COMPLETE")
print("="*70)
print("Files created:")
print("  - results/et_distill_pilot500_cond55.csv")
print("  - results/et_distill_pilot500_cond55_pred.csv")
print("  - results/et_distill_pilot500_cond80.csv")
print("  - results/et_distill_pilot500_cond80_pred.csv")
