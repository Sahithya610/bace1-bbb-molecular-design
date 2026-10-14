"""
Lambda-condition sweep: 4 × 500-step pilots at λ_condition ∈ {1, 5, 10, 20}.
All other settings identical to the ET-distill Pilot500 baseline.

For each λ this script:
  1. Trains 500 steps warm-started from Issue3-ChemistryReadout.
  2. Generates 1024 molecules at pIC50=5.5 and pIC50=8.0.
  3. Predicts pIC50/logBB with predict_joint.
  4. Reports Δμ(8.0−5.5) for that λ.

Results are printed and written to lambda_sweep_results.txt.
"""
import argparse
import os
import sys
import traceback
import subprocess
import pandas as np_  # shadowed below
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

LAMBDAS   = [1.0, 5.0, 10.0, 20.0]
BASE_GEN  = "experiments/models/Issue3-ChemistryReadout/joint_generator.pt"
BASE_DIS  = "experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt"
ET_LABELS = "druggen_data/bace1_et_predictions.csv"
RESULT_CSV = "lambda_sweep_results.csv"
RESULT_TXT = "lambda_sweep_results.txt"

SEP = "=" * 72

class Tee:
    def __init__(self, *targets): self.targets = targets
    def write(self, s):
        for t in self.targets: t.write(s); t.flush()
        return len(s)
    def flush(self):
        for t in self.targets: t.flush()

log_f = open(RESULT_TXT, "w", encoding="utf-8", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_f)
sys.stderr = sys.stdout

print(SEP)
print("LAMBDA_CONDITION SWEEP  (500 steps, ET-distill, warm-start)")
print(SEP)
print(f"Lambdas: {LAMBDAS}")
print(f"Baseline: {BASE_GEN}")
print()

rows = []

for lam in LAMBDAS:
    tag = f"lam{lam:.0f}".replace(".", "_")
    out_dir  = f"experiments/models/Issue3-LambdaSweep-{tag}"
    gen_lo   = f"results/lambda_sweep_{tag}_cond55.csv"
    gen_hi   = f"results/lambda_sweep_{tag}_cond80.csv"
    pred_lo  = f"results/lambda_sweep_{tag}_cond55_pred.csv"
    pred_hi  = f"results/lambda_sweep_{tag}_cond80_pred.csv"
    log_path = f"lambda_sweep_{tag}_log.txt"

    print(f"\n{SEP}")
    print(f"λ_condition = {lam}  →  {out_dir}")
    print(SEP)

    # ── Step 1: train ────────────────────────────────────────────────────
    print("Training 500 steps …")
    try:
        import train_joint_et_distill as tjed
        cfg = argparse.Namespace(
            chembl_smi="data/chembl_train.smi",
            bace1_smi="druggen_data/bace1.smi",
            labels_csv="druggen_data/bace1_continuous_bbb.csv",
            et_labels=ET_LABELS,
            data_dir="data",
            output_dir=out_dir,
            init_generator=BASE_GEN,
            init_discriminator=BASE_DIS,
            max_atom=45,
            batch_size=32,
            epochs=1,
            max_steps=500,
            dim=128, depth=1, ddepth=1, heads=8, mlp_ratio=3,
            dropout=0.0, ddropout=0.0,
            g_lr=1e-5, d_lr=1e-5, beta1=0.9, beta2=0.999,
            lambda_gp=10.0,
            lambda_aux=1.0,
            lambda_condition=lam,
            property_input="st",
            aux_holdout_frac=0.1,
            act="relu",
            seed=1,
        )

        # Redirect output for this run to both console and per-run log
        per_log = open(log_path, "w", encoding="utf-8", buffering=1)
        old_stdout = sys.stdout
        sys.stdout = Tee(old_stdout, per_log)
        try:
            tjed.train(cfg)
        finally:
            sys.stdout = old_stdout
            per_log.close()

        ckpt = f"{out_dir}/joint_generator.pt"
        print(f"  ✓ Checkpoint saved: {ckpt}")

    except Exception as e:
        traceback.print_exc()
        print(f"  ✗ Training FAILED: {e}")
        rows.append({"lambda": lam, "error": str(e)})
        continue

    # ── Step 2: generate at two conditions ───────────────────────────────
    py = sys.executable
    for cond, out_csv in [(5.5, gen_lo), (8.0, gen_hi)]:
        print(f"  Generating at pIC50={cond} …")
        r = subprocess.run([
            py, "generate_joint_molecules.py",
            "--checkpoint", ckpt,
            "--desired_pIC50", str(cond),
            "--desired_logBB", "0.0",
            "--batch_size", "128",
            "--num_batches", "8",
            "--seed", "42",
            "--output", out_csv,
        ], capture_output=True, text=True,
           cwd=os.path.dirname(os.path.abspath(__file__)))
        if r.returncode != 0:
            print(f"    ✗ Generation failed (code {r.returncode})")
            print(r.stderr[-800:] if r.stderr else "")
        else:
            n = sum(1 for _ in open(out_csv)) - 1
            print(f"    ✓ {n} molecules → {out_csv}")

    # ── Step 3: predict ───────────────────────────────────────────────────
    for in_csv, out_csv in [(gen_lo, pred_lo), (gen_hi, pred_hi)]:
        if not os.path.exists(in_csv):
            print(f"  ✗ Skipping prediction (missing {in_csv})")
            continue
        print(f"  Predicting {in_csv} …")
        r = subprocess.run([
            py, "predict_joint.py",
            "--input", in_csv,
            "--output", out_csv,
        ], capture_output=True, text=True,
           cwd=os.path.dirname(os.path.abspath(__file__)))
        if r.returncode != 0:
            print(f"    ✗ Prediction failed (code {r.returncode})")
            print(r.stderr[-400:] if r.stderr else "")
        else:
            n = sum(1 for _ in open(out_csv)) - 1
            print(f"    ✓ {n} predictions → {out_csv}")

    # ── Step 4: measure Δ ─────────────────────────────────────────────────
    try:
        lo = pd.read_csv(pred_lo)["predicted_pIC50"]
        hi = pd.read_csv(pred_hi)["predicted_pIC50"]
        dm   = float(hi.mean() - lo.mean())
        dmed = float(hi.median() - lo.median())
        lo_mean = float(lo.mean()); lo_n = len(lo)
        hi_mean = float(hi.mean()); hi_n = len(hi)
        flag = "✓ PASS" if abs(dm) >= 0.3 else "✗ FAIL"

        print(f"\n  λ={lam:4.0f}  "
              f"μ_lo={lo_mean:.3f}(n={lo_n})  "
              f"μ_hi={hi_mean:.3f}(n={hi_n})  "
              f"Δμ={dm:+.3f}  {flag}")

        rows.append({
            "lambda_condition": lam,
            "n_lo": lo_n, "mean_lo": lo_mean,
            "n_hi": hi_n, "mean_hi": hi_mean,
            "delta_mean": dm, "delta_median": dmed,
            "pass": abs(dm) >= 0.3,
            "checkpoint": ckpt,
        })
    except Exception as e:
        print(f"  ✗ Δ computation failed: {e}")
        rows.append({"lambda_condition": lam, "error": str(e)})

# ── Final summary table ───────────────────────────────────────────────────
print(f"\n\n{SEP}")
print("SWEEP SUMMARY")
print(SEP)
print(f"  {'λ_cond':>8}  {'n_lo':>6}  {'μ_lo':>6}  {'n_hi':>6}  "
      f"{'μ_hi':>6}  {'Δμ':>7}  {'Result':>8}")
print("  " + "-"*60)
for row in rows:
    if "error" in row:
        print(f"  {row.get('lambda_condition', '?'):>8}  ERROR: {row['error']}")
    else:
        flag = "PASS ✓" if row["pass"] else "FAIL ✗"
        print(f"  {row['lambda_condition']:>8.1f}  "
              f"{row['n_lo']:>6}  {row['mean_lo']:>6.3f}  "
              f"{row['n_hi']:>6}  {row['mean_hi']:>6.3f}  "
              f"{row['delta_mean']:>+7.3f}  {flag:>8}")

# Save CSV
if rows:
    pd.DataFrame(rows).to_csv(RESULT_CSV, index=False)
    print(f"\nFull results: {RESULT_CSV}")

print(f"Log:          {RESULT_TXT}")

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_f.close()
