"""Thin wrapper: invokes predict_joint main() with sys.argv patched."""
import sys
sys.argv = [
    "predict_joint.py",
    "--input",  "results/baseline_screen_50000_generated.csv",
    "--output", "results/baseline_screen_50000_predictions.csv",
]

# Change to project root so relative paths in predict_joint resolve correctly
import os
os.chdir(r"c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master")

# Run predict_joint's main
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "predict_joint",
    r"c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\predict_joint.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.main()
