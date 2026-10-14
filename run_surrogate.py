"""Run surrogate training as a subprocess and stream output to log + stdout."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent
python = root / ".venv" / "Scripts" / "python.exe"
log_path = root / "surrogate_training_log.txt"

cmd = [
    str(python),
    str(root / "train_bace1_surrogate.py"),
    "--input",      "druggen_data/bace1_continuous_bbb.csv",
    "--output_dir", "results/bace1_surrogate_model",
    "--epochs",     "200",
]

print("Running:", " ".join(cmd))
print("Log:", log_path)

with open(log_path, "w", encoding="utf-8") as log_f:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(root),
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_f.write(line)
        log_f.flush()

proc.wait()
print(f"\nProcess exited with code {proc.returncode}")
sys.exit(proc.returncode)
