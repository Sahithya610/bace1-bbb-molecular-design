"""
Run the surrogate pilot v2 and write all output to pilot_v2_log.txt.
Usage:  .venv\Scripts\python.exe run_pilot_v2.py
"""
import argparse
import sys
import os
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))

LOG = "pilot_v3_log.txt"

class Logger:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.__stdout__
        self._stderr = sys.__stderr__
    def write(self, s):
        self.f.write(s); self.f.flush()
        self._stdout.write(s); self._stdout.flush()
        return len(s)
    def flush(self):
        self.f.flush(); self._stdout.flush()

log = Logger(LOG)
sys.stdout = log
sys.stderr = log

print("=== Surrogate Pilot v3 (lambda_surrogate=0.05, relative scaling + clamp=0.5) ===")

cfg = argparse.Namespace(
    chembl_smi="data/chembl_train.smi",
    bace1_smi="druggen_data/bace1.smi",
    labels_csv="druggen_data/bace1_continuous_bbb.csv",
    surrogate="results/bace1_surrogate_model/bace1_surrogate.pt",
    data_dir="data",
    output_dir="experiments/models/Issue3-SurrogateLoss-Pilot100v3",
    init_generator="experiments/models/Issue3-ChemistryReadout/joint_generator.pt",
    init_discriminator="experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt",
    max_atom=45,
    batch_size=32,
    epochs=1,
    max_steps=100,
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
    lambda_surrogate=0.05,
    property_input="st",
    aux_holdout_frac=0.1,
    act="relu",
    seed=1,
)

try:
    import train_joint_surrogate as tjs
    tjs.train(cfg)
    print("\n=== PILOT COMPLETE ===")
except Exception:
    traceback.print_exc()
    print("\n=== PILOT FAILED ===")
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log.f.close()

print(f"Log written to: {LOG}")
