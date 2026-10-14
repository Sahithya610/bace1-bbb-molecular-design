"""
REINFORCE oracle pilot — 500 steps, λ_oracle=5.0, warm-start from baseline.
Writes log to reinforce_pilot_log.txt.
"""
import argparse, os, sys, traceback, subprocess
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

LOG = "reinforce_pilot_v4_log.txt"

class Tee:
    def __init__(self, *targets): self.targets = targets
    def write(self, s):
        for t in self.targets: t.write(s); t.flush()
        return len(s)
    def flush(self):
        for t in self.targets: t.flush()

log_f = open(LOG, "w", encoding="utf-8", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_f)
sys.stderr = sys.stdout

try:
    import train_joint_reinforce as tjr

    cfg = argparse.Namespace(
        chembl_smi="data/chembl_train.smi",
        bace1_smi="druggen_data/bace1.smi",
        labels_csv="druggen_data/bace1_continuous_bbb.csv",
        data_dir="data",
        output_dir="experiments/models/Issue3-Reinforce-Pilot500v4",
        init_generator="experiments/models/Issue3-ChemistryReadout/joint_generator.pt",
        init_discriminator="experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt",
        max_atom=45,
        batch_size=32,
        epochs=1,
        max_steps=500,
        dim=128, depth=1, ddepth=1, heads=8, mlp_ratio=3,
        dropout=0.0, ddropout=0.0,
        g_lr=1e-5, d_lr=1e-5, beta1=0.9, beta2=0.999,
        lambda_gp=10.0,
        lambda_oracle=5.0,
        act="relu",
        seed=1,
    )

    tjr.train(cfg)
    print("\nTRAINING DONE")

except Exception:
    traceback.print_exc()
    print("\n=== TRAINING FAILED ===")
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_f.close()

# ── Generate & evaluate condition sensitivity ──────────────────────────
py = sys.executable
ckpt = "experiments/models/Issue3-Reinforce-Pilot500v4/joint_generator.pt"

for cond, out in [(5.5, "results/reinforce_pilot_v4_cond55.csv"),
                  (8.0, "results/reinforce_pilot_v4_cond80.csv")]:
    subprocess.run([
        py, "generate_joint_molecules.py",
        "--checkpoint", ckpt,
        "--desired_pIC50", str(cond), "--desired_logBB", "0.0",
        "--batch_size", "128", "--num_batches", "8",
        "--seed", "42", "--output", out,
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

for in_csv, out_csv in [
    ("results/reinforce_pilot_v4_cond55.csv", "results/reinforce_pilot_v4_cond55_pred.csv"),
    ("results/reinforce_pilot_v4_cond80.csv", "results/reinforce_pilot_v4_cond80_pred.csv"),
]:
    subprocess.run([py, "predict_joint.py", "--input", in_csv, "--output", out_csv],
                   cwd=os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*60)
print("REINFORCE v4 PILOT (contrastive batches, soft reward)")
print("="*60)
try:
    lo = pd.read_csv("results/reinforce_pilot_v4_cond55_pred.csv")["predicted_pIC50"]
    hi = pd.read_csv("results/reinforce_pilot_v4_cond80_pred.csv")["predicted_pIC50"]
    dm = float(hi.mean() - lo.mean())
    print(f"μ_lo={lo.mean():.3f} (n={len(lo)})  μ_hi={hi.mean():.3f} (n={len(hi)})  Δμ={dm:+.3f}")
    print(f"{'PASS ✓' if abs(dm)>=0.3 and len(lo)>=10 else 'FAIL ✗'}  (need |Δ|≥0.3 and n_lo≥10)")
except Exception as e:
    print(f"Δ computation: {e}")
