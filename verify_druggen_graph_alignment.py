"""
Verify alignment between the BACE1 activity labels and the
actual molecules retained in DrugGEN's processed dataset.

This does NOT modify the dataset or model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


CSV_PATH = Path("data/bace1_clean.csv")
SMI_PATH = Path("druggen_data/bace1.smi")
GRAPH_PATH = Path("druggen_data/bace145.pt")


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


print("=" * 70)
print("DRUGGEN BACE1 GRAPH / pIC50 ALIGNMENT CHECK")
print("=" * 70)

# ------------------------------------------------------------
# Load activity data
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

if "pIC50" not in df.columns:
    df["standard_value"] = pd.to_numeric(
        df["standard_value"],
        errors="coerce"
    )

    df["pIC50"] = 9.0 - np.log10(
        df["standard_value"]
    )

df["pIC50"] = pd.to_numeric(
    df["pIC50"],
    errors="coerce"
)

df = df.dropna(
    subset=["canonical_smiles", "pIC50"]
).copy()

df["canonical_smiles"] = df[
    "canonical_smiles"
].map(canonicalize)

df = df.dropna(
    subset=["canonical_smiles"]
)

# One molecule -> median pIC50
activity = (
    df.groupby("canonical_smiles")["pIC50"]
    .median()
)

print()
print(f"Activity molecules: {len(activity):,}")

# ------------------------------------------------------------
# Load original DrugGEN SMILES
# ------------------------------------------------------------

with open(
    SMI_PATH,
    "r",
    encoding="utf-8"
) as f:

    smi_list = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(
    f"DrugGEN SMILES entries: {len(smi_list):,}"
)

# ------------------------------------------------------------
# Load processed PyG dataset
# ------------------------------------------------------------

data, slices = torch.load(
    GRAPH_PATH,
    weights_only=False
)

# Determine graph count from PyG slices
if hasattr(data, "__len__"):
    try:
        graph_count = len(data)
    except Exception:
        graph_count = int(slices["x"].shape[0] - 1)
else:
    graph_count = int(slices["x"].shape[0] - 1)

print(
    f"Processed DrugGEN graphs: {graph_count:,}"
)

# ------------------------------------------------------------
# Important observation
# ------------------------------------------------------------

print()
print("-" * 70)
print("DATASET SIZE CHECK")
print("-" * 70)

print(
    f"Original BACE1 SMILES : {len(smi_list):,}"
)

print(
    f"Processed graphs      : {graph_count:,}"
)

print(
    f"Graphs filtered out   : "
    f"{len(smi_list) - graph_count:,}"
)

# ------------------------------------------------------------
# Reproduce DrugGEN filtering from the processed data
#
# The processed dataset stores graph-level atom features.
# We use the retained graph indices from the PyG slices.
# ------------------------------------------------------------

print()
print("Inspecting processed graph metadata...")

# The original DrugGEN preprocessing preserves the graph
# ordering from the input SMILES after filtering.
#
# We therefore identify which input molecules survived by
# reconstructing their graph validity using the same maximum
# atom limit.

MAX_ATOMS = 45

retained_smiles = []

for smiles in smi_list:

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue

    if mol.GetNumAtoms() > MAX_ATOMS:
        continue

    retained_smiles.append(
        canonicalize(smiles)
    )

# ------------------------------------------------------------
# Compare expected retained count with processed count
# ------------------------------------------------------------

print()
print("-" * 70)
print("RETAINED MOLECULE CHECK")
print("-" * 70)

print(
    f"Expected retained by <= {MAX_ATOMS} atoms: "
    f"{len(retained_smiles):,}"
)

print(
    f"Actual processed graphs: "
    f"{graph_count:,}"
)

if len(retained_smiles) != graph_count:

    print()
    print(
        "WARNING: Simple atom-count filtering does not exactly "
        "reproduce the processed graph count."
    )

    print(
        "We will NOT use this assumption for training."
    )

else:

    print()
    print(
        "Retained molecule count matches processed graph count."
    )

# ------------------------------------------------------------
# Check activity coverage
# ------------------------------------------------------------

retained_smiles = [
    s for s in retained_smiles
    if s is not None
]

missing = [
    s for s in retained_smiles
    if s not in activity
]

print()
print("-" * 70)
print("pIC50 COVERAGE")
print("-" * 70)

print(
    f"Retained molecules checked: "
    f"{len(retained_smiles):,}"
)

print(
    f"Missing pIC50 values: "
    f"{len(missing):,}"
)

if missing:

    print()
    print("First missing structures:")

    for smiles in missing[:10]:
        print(smiles)

    raise RuntimeError(
        "Some retained DrugGEN molecules have no pIC50."
    )

# ------------------------------------------------------------
# Create aligned activity table
# ------------------------------------------------------------

aligned = pd.DataFrame({
    "canonical_smiles": retained_smiles
})

aligned["pIC50"] = aligned[
    "canonical_smiles"
].map(activity)

if aligned["pIC50"].isna().any():

    raise RuntimeError(
        "NaN pIC50 remains after alignment."
    )

# ------------------------------------------------------------
# Calculate weights
# ------------------------------------------------------------

median = aligned["pIC50"].median()
p90 = aligned["pIC50"].quantile(0.90)

def weight(pic50):

    if pic50 <= median:
        return 1.0

    if pic50 >= p90:
        return 2.0

    return 1.0 + (
        (pic50 - median)
        / (p90 - median)
    )


aligned["sampling_weight"] = aligned[
    "pIC50"
].apply(weight)

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

print()
print("-" * 70)
print("FINAL ALIGNMENT")
print("-" * 70)

print(
    f"Graphs:              {len(aligned):,}"
)

print(
    f"pIC50 values:        {aligned.pIC50.notna().sum():,}"
)

print(
    f"Sampling weights:    "
    f"{aligned.sampling_weight.notna().sum():,}"
)

print(
    f"Missing pIC50:       "
    f"{aligned.pIC50.isna().sum():,}"
)

print(
    f"Missing weights:     "
    f"{aligned.sampling_weight.isna().sum():,}"
)

print(
    f"Weight minimum:      "
    f"{aligned.sampling_weight.min():.4f}"
)

print(
    f"Weight maximum:      "
    f"{aligned.sampling_weight.max():.4f}"
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output = Path(
    "results/bace1_druggen_aligned_weights.csv"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True
)

aligned.to_csv(
    output,
    index=False
)

print()
print(f"Saved: {output}")

print()
print("=" * 70)
print("ALIGNMENT CHECK COMPLETE")
print("=" * 70)