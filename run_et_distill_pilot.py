"""
Run ET-distill pipeline steps 1 and 2, log to et_distill_pilot_log.txt.

Step 1: precompute ExtraTrees pIC50 predictions on training molecules.
Step 2: 100-step pilot warm-started from the baseline checkpoint.

After this script completes, run the condition sensitivity test manually:

    .venv\\Scripts\\python.exe generate_joint_molecules.py
        --checkpoint experiments/models/Issue3-ETDistill-Pilot100/joint_generator.pt
        --desired_pIC50 5.5 --desired_logBB 0.0
        --batch_size 128 --num_batches 8 --seed 42
        --output results/et_distill_pilot_cond55.csv

    .venv\\Scripts\\python.exe generate_joint_molecules.py
        --checkpoint experiments/models/Issue3-ETDistill-Pilot100/joint_generator.pt
        --desired_pIC50 8.0 --desired_logBB 0.0
        --batch_size 128 --num_batches 8 --seed 42
        --output results/et_distill_pilot_cond80.csv

    .venv\\Scripts\\python.exe predict_joint.py
        --input results/et_distill_pilot_cond55.csv
        --output results/et_distill_pilot_cond55_pred.csv

    .venv\\Scripts\\python.exe predict_joint.py
        --input results/et_distill_pilot_cond80.csv
        --output results/et_distill_pilot_cond80_pred.csv
"""
import argparse
import os
import sys
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))
LOG = "et_distill_pilot500_log.txt"


class Tee:
    def __init__(self, *targets):
        self.targets = targets
    def write(self, s):
        for t in self.targets:
            t.write(s)
            t.flush()
        return len(s)
    def flush(self):
        for t in self.targets:
            t.flush()


log_f = open(LOG, "w", encoding="utf-8", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_f)
sys.stderr = sys.stdout

try:
    # ====================================================
    # Step 1: precompute ET labels (skip if already done)
    # ====================================================
    import os
    et_csv = "druggen_data/bace1_et_predictions.csv"
    if not os.path.exists(et_csv):
        print("\n" + "=" * 70)
        print("STEP 1: Precompute ExtraTrees pIC50 predictions")
        print("=" * 70)
        import precompute_et_labels
        precompute_et_labels.main()
    else:
        print(f"STEP 1: ET predictions already exist at {et_csv}, skipping.")

    # ====================================================
    # Step 2: 100-step ET-distill pilot
    # ====================================================
    print("\n" + "=" * 70)
    print("STEP 2: ET-distill pilot (500 steps, batch=32, warm-start)")
    print("=" * 70)

    cfg = argparse.Namespace(
        chembl_smi="data/chembl_train.smi",
        bace1_smi="druggen_data/bace1.smi",
        labels_csv="druggen_data/bace1_continuous_bbb.csv",
        et_labels="druggen_data/bace1_et_predictions.csv",
        data_dir="data",
        output_dir="experiments/models/Issue3-ETDistill-Pilot500",
        init_generator=(
            "experiments/models/Issue3-ChemistryReadout/joint_generator.pt"
        ),
        init_discriminator=(
            "experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt"
        ),
        max_atom=45,
        batch_size=32,
        epochs=1,
        max_steps=500,
        dim=128,
        depth=1,
        ddepth=1,
        heads=8,
        mlp_ratio=3,
        dropout=0.0,
        ddropout=0.0,
        g_lr=1e-5,
        d_lr=1e-5,
        beta1=0.9,
        beta2=0.999,
        lambda_gp=10.0,
        lambda_aux=1.0,
        lambda_condition=1.0,
        property_input="st",
        aux_holdout_frac=0.1,
        act="relu",
        seed=1,
    )

    import train_joint_et_distill as tjed
    tjed.train(cfg)

    print("\n" + "=" * 70)
    print("Steps 1 and 2 complete.")
    print("Run the generation + evaluation commands from the script docstring.")
    print("=" * 70)

except Exception:
    traceback.print_exc()
    print("\n=== PIPELINE FAILED ===")
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_f.close()

print(f"Log written to: {LOG}")
