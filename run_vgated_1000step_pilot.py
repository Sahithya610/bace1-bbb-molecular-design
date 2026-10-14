# File: run_vgated_1000step_pilot.py
# Run: .venv\Scripts\python.exe run_vgated_1000step_pilot.py
import argparse, os, sys, traceback, subprocess
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = "experiments/models/Issue3-VGated2-1000step"
LOG        = "vgated_1000step_log.txt"

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
    import train_joint_et_distill_vgated as tjed

    cfg = argparse.Namespace(
        chembl_smi="data/chembl_train.smi",
        bace1_smi="druggen_data/bace1.smi",
        labels_csv="druggen_data/bace1_continuous_bbb.csv",
        et_labels="druggen_data/bace1_et_predictions.csv",
        data_dir="data",
        output_dir=OUTPUT_DIR,
        init_generator="experiments/models/Issue3-ChemistryReadout/joint_generator.pt",
        init_discriminator="experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt",
        max_atom=45,
        batch_size=32,
        epochs=1,
        max_steps=1000,          # doubled from 500-step pilot
        dim=128, depth=1, ddepth=1, heads=8, mlp_ratio=3,
        dropout=0.0, ddropout=0.0,
        g_lr=1e-5, d_lr=1e-5, beta1=0.9, beta2=0.999,
        lambda_gp=10.0,
        lambda_aux=1.0,
        lambda_condition=5.0,    # winning config
        lambda_padding=5.0,      # winning config
        property_input="st",
        aux_holdout_frac=0.1,
        act="relu",
        seed=1,
    )

    tjed.train(cfg)
    print("\nTRAINING DONE")

except Exception:
    traceback.print_exc()
    print("\n=== TRAINING FAILED ===")
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_f.close()

# --- Generate at two conditions ---
py = sys.executable
ckpt = f"{OUTPUT_DIR}/joint_generator.pt"

print("\nGenerating molecules at pIC50=5.5 and pIC50=8.0 ...")
for cond, out in [(5.5, "results/vg_1000step_cond55.csv"),
                   (8.0, "results/vg_1000step_cond80.csv")]:
    print(f"  pIC50={cond} ...")
    subprocess.run([
        py, "generate_joint_molecules.py",
        "--checkpoint", ckpt,
        "--desired_pIC50", str(cond),
        "--desired_logBB", "0.0",
        "--batch_size", "128",
        "--num_batches", "8",
        "--seed", "42",
        "--output", out,
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

# --- Predict ---
print("\nRunning oracle predictions ...")
for in_csv, out_csv in [
    ("results/vg_1000step_cond55.csv", "results/vg_1000step_cond55_pred.csv"),
    ("results/vg_1000step_cond80.csv", "results/vg_1000step_cond80_pred.csv"),
]:
    print(f"  {in_csv} ...")
    subprocess.run([py, "predict_joint.py", "--input", in_csv, "--output", out_csv],
                   cwd=os.path.dirname(os.path.abspath(__file__)))

# --- Report Δ ---
print("\n" + "="*60)
print("1000-STEP PILOT RESULT")
print("="*60)
try:
    lo = pd.read_csv("results/vg_1000step_cond55_pred.csv")["predicted_pIC50"]
    hi = pd.read_csv("results/vg_1000step_cond80_pred.csv")["predicted_pIC50"]
    dm = float(hi.mean() - lo.mean())
    print(f"μ_lo = {lo.mean():.3f}  (n={len(lo)})  [pIC50=5.5 requested]")
    print(f"μ_hi = {hi.mean():.3f}  (n={len(hi)})  [pIC50=8.0 requested]")
    print(f"Δμ   = {dm:+.3f}")
    print()
    if abs(dm) >= 0.3 and len(lo) >= 50:
        print("DECISION: ✓ PROCEED TO FULL TRAINING")
        print("  Run train_joint_et_distill_vgated.py for 10 epochs,")
        print("  batch=128, lambda_condition=5, lambda_padding=5,")
        print("  warm-started from Issue3-ChemistryReadout.")
    elif abs(dm) >= 0.3 and len(lo) < 50:
        print("DECISION: ⚠ Δ passes but n_lo is small — check oracle filter.")
        print("  Investigate why predict_joint.py drops most low-condition molecules.")
        print("  The Δ signal is likely real; verify before full training.")
    else:
        print("DECISION: ✗ Δ < 0.3 — try lambda_padding=10 for another 500-step pilot.")
except Exception as e:
    print(f"Could not compute Δ: {e}")
    print("Check results/vg_1000step_cond55_pred.csv and vg_1000step_cond80_pred.csv manually.")
