"""
Self-contained surrogate training runner.
Writes all output to surrogate_training_log.txt with line-by-line flushing.
Run with:
    .venv\Scripts\python.exe run_and_log.py
"""
import io
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
log_path = "surrogate_training_log.txt"

class Tee(io.TextIOBase):
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

log_f = open(log_path, "w", encoding="utf-8", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_f)
sys.stderr = sys.stdout

import argparse
import train_bace1_surrogate as tbs

# Build a minimal namespace directly — no argparse needed.
cfg = argparse.Namespace(
    input="druggen_data/bace1_continuous_bbb.csv",
    output_dir="results/bace1_surrogate_model",
    epochs=200,
    batch_size=256,
    lr=1e-3,
    dropout=0.2,
)

try:
    r2 = tbs.train(cfg)
    print(f"\nFinal test R²: {r2:.4f}")
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    log_f.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
