"""
Validity-gated + padding-penalty sweep.
Tests λ_condition ∈ {5, 10} × λ_padding ∈ {1, 5} with the vgated script.

The padding penalty rewards the generator for keeping atom-class-0 (padding)
slots, directly counteracting the saturation that caused 0% validity at
pIC50=8.0 when lambda_condition was amplified.

For each config this script:
  1. Trains 500 steps warm-started from Issue3-ChemistryReadout.
  2. Generates 1024 molecules at pIC50=5.5 and pIC50=8.0.
  3. Predicts pIC50/logBB and reports Δμ(8.0−5.5).

Results written to vgated_sweep2_results.txt / .csv.
"""
import argparse, os, sys, traceback, subprocess
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# (lambda_condition, lambda_padding)
CONFIGS   = [(5.0, 1.0), (5.0, 5.0), (10.0, 1.0), (10.0, 5.0)]
BASE_GEN  = "experiments/models/Issue3-ChemistryReadout/joint_generator.pt"
BASE_DIS  = "experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt"
ET_LABELS = "druggen_data/bace1_et_predictions.csv"
RESULT_CSV = "vgated_sweep2_results.csv"
RESULT_TXT = "vgated_sweep2_results.txt"

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
print("VALIDITY-GATED + PADDING-PENALTY SWEEP  (500 steps, ET-distill)")
print(SEP)
print(f"Configs (λ_cond, λ_pad): {CONFIGS}")
print()

rows = []

for lam_c, lam_p in CONFIGS:
    tag     = f"vg2_c{lam_c:.0f}_p{lam_p:.0f}"
    out_dir = f"experiments/models/Issue3-VGated2-{tag}"
    gen_lo  = f"results/{tag}_cond55.csv"
    gen_hi  = f"results/{tag}_cond80.csv"
    pred_lo = f"results/{tag}_cond55_pred.csv"
    pred_hi = f"results/{tag}_cond80_pred.csv"
    log_path = f"vgated2_{tag}_log.txt"

    print(f"\n{SEP}")
    print(f"λ_condition={lam_c}  λ_padding={lam_p}  →  {out_dir}")
    print(SEP)

    # ── Train ─────────────────────────────────────────────────────────
    print("Training 500 steps …")
    try:
        import train_joint_et_distill_vgated as tjed_vg
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
            lambda_condition=lam_c,
            lambda_padding=lam_p,
            property_input="st",
            aux_holdout_frac=0.1,
            act="relu",
            seed=1,
        )

        per_log = open(log_path, "w", encoding="utf-8", buffering=1)
        old_stdout = sys.stdout
        sys.stdout = Tee(old_stdout, per_log)
        try:
            tjed_vg.train(cfg)
        finally:
            sys.stdout = old_stdout
            per_log.close()

        ckpt = f"{out_dir}/joint_generator.pt"
        print(f"  ✓ Checkpoint saved: {ckpt}")

    except Exception as e:
        traceback.print_exc()
        print(f"  ✗ Training FAILED: {e}")
        rows.append({"lambda_condition": lam_c, "lambda_padding": lam_p, "error": str(e)})
        continue

    # ── Generate ──────────────────────────────────────────────────────
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
            print(f"    ✗ Generation failed: {r.stderr[-300:]}")
        else:
            try:
                n_valid = sum(1 for l in open(out_csv) if ",True," in l)
                n_total = sum(1 for _ in open(out_csv)) - 1
                print(f"    ✓ {n_valid}/{n_total} valid → {out_csv}")
            except Exception:
                print(f"    ✓ done → {out_csv}")

    # ── Predict ───────────────────────────────────────────────────────
    for in_csv, out_csv in [(gen_lo, pred_lo), (gen_hi, pred_hi)]:
        if not os.path.exists(in_csv):
            print(f"  ✗ Skipping (missing {in_csv})")
            continue
        print(f"  Predicting {in_csv} …")
        r = subprocess.run([
            py, "predict_joint.py",
            "--input", in_csv,
            "--output", out_csv,
        ], capture_output=True, text=True,
           cwd=os.path.dirname(os.path.abspath(__file__)))
        if r.returncode != 0:
            print(f"    ✗ Prediction failed — {r.stderr[-200:]}")
        else:
            n = sum(1 for _ in open(out_csv)) - 1 if os.path.exists(out_csv) else 0
            print(f"    ✓ {n} predictions → {out_csv}")

    # ── Measure Δ ─────────────────────────────────────────────────────
    row = {"lambda_condition": lam_c, "lambda_padding": lam_p}
    try:
        lo = pd.read_csv(pred_lo)["predicted_pIC50"]
        hi = pd.read_csv(pred_hi)["predicted_pIC50"]
        dm   = float(hi.mean() - lo.mean())
        dmed = float(hi.median() - lo.median())
        flag = "✓ PASS" if abs(dm) >= 0.3 else "✗ FAIL"
        print(f"\n  λ_c={lam_c} λ_p={lam_p}  "
              f"μ_lo={lo.mean():.3f}(n={len(lo)})  "
              f"μ_hi={hi.mean():.3f}(n={len(hi)})  "
              f"Δμ={dm:+.3f}  {flag}")
        row.update({"n_lo": len(lo), "mean_lo": float(lo.mean()),
                    "n_hi": len(hi), "mean_hi": float(hi.mean()),
                    "delta_mean": dm, "delta_median": dmed,
                    "pass": abs(dm) >= 0.3})
    except Exception as e:
        print(f"  ✗ Δ computation failed: {e}")
        row["error"] = str(e)
    rows.append(row)

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n\n{SEP}")
print("SWEEP SUMMARY")
print(SEP)
print(f"  {'λ_c':>5}  {'λ_p':>5}  {'n_lo':>6}  {'μ_lo':>6}  "
      f"{'n_hi':>6}  {'μ_hi':>6}  {'Δμ':>7}  Result")
print("  " + "-"*60)
print(f"  {'1.0':>5}  {'—':>5}  {'189':>6}  {'5.409':>6}  "
      f"{'305':>6}  {'5.475':>6}  {'+0.066':>7}  FAIL ✗  (λ=1 baseline)")
for row in rows:
    if "error" in row:
        print(f"  {row['lambda_condition']:>5.1f}  {row['lambda_padding']:>5.1f}  ERROR: {row['error']}")
    else:
        flag = "PASS ✓" if row.get("pass") else "FAIL ✗"
        print(f"  {row['lambda_condition']:>5.1f}  {row['lambda_padding']:>5.1f}  "
              f"{row['n_lo']:>6}  {row['mean_lo']:>6.3f}  "
              f"{row['n_hi']:>6}  {row['mean_hi']:>6.3f}  "
              f"{row['delta_mean']:>+7.3f}  {flag}")

if rows:
    pd.DataFrame(rows).to_csv(RESULT_CSV, index=False)
    print(f"\nResults: {RESULT_CSV}")
print(f"Log:     {RESULT_TXT}")

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_f.close()
