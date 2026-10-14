import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
exe = shutil.which("mk_prepare_receptor") or str(ROOT / ".venv" / "Scripts" / "mk_prepare_receptor.exe")

r = subprocess.run(
    [exe,
     "-i", "docking/bace1_receptor.pdb",
     "-o", "docking/bace1_receptor",
     "--compute_charges", "--charge_model", "gasteiger"],
    capture_output=True, text=True, cwd=str(ROOT)
)
print("stdout:", r.stdout[:1000])
print("stderr:", r.stderr[:1000])
print("code:", r.returncode)
import os
print("exists:", os.path.exists("docking/bace1_receptor.pdbqt"))
if os.path.exists("docking/bace1_receptor.pdbqt"):
    with open("docking/bace1_receptor.pdbqt") as f:
        lines = f.readlines()
    print(f"Lines: {len(lines)}")
    print("First 3:", lines[:3])
