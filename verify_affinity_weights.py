"""
Verify mild affinity-aware sampling weights against the actual
BACE1 molecules retained by DrugGEN preprocessing.

IMPORTANT:
- Does NOT modify the dataset.
- Does NOT modify the model.
- Does NOT train anything.
- Every retained DrugGEN molecule must receive exactly one pIC50
  and exactly one sampling weight.

Weight design:
    pIC50 <= median  -> weight 1.0
    median < pIC50 < 90th percentile
        -> smoothly increase from 1.0 to 2.0
    pIC50 >= 90th percentile -> weight 2.0

Maximum weight = 2.0.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


BACE1_CSV = Path("data/bace1_clean.csv")
BACE1_SMI = Path("druggen_data/bace1.smi")

OUTPUT = Path(
    "results/bace1_affinity_weight_verification.csv"
)


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


print("=" * 70)
print("BACE1 AFFINITY WEIGHT VERIFICATION")
print("=" * 70)

# ------------------------------------------------------------
# Load cleaned BACE1 activity data
# ------------------------------------------------------------

df = pd.read_csv(BACE1_CSV)

print(f"CSV records: {len(df):,}")

if "pIC50" not in df.columns:

    if "standard_value" not in df.columns:
        raise ValueError(
            "Neither pIC50 nor standard_value exists."
        )

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

# ------------------------------------------------------------
# Canonicalize and obtain one molecule-level pIC50
# ------------------------------------------------------------

df["canonical_smiles"] = df["canonical_smiles"].map(
    canonicalize
)

df = df.dropna(
    subset=["canonical_smiles"]
).copy()

molecule_activity = (
    df.groupby("canonical_smiles")["pIC50"]
    .median()
)

print(
    f"Unique canonical molecules in CSV: "
    f"{len(molecule_activity):,}"
)

# ------------------------------------------------------------
# Load the EXACT molecules used by DrugGEN preprocessing
# ------------------------------------------------------------

with open(
    BACE1_SMI,
    "r",
    encoding="utf-8"
) as f:

    druggen_smiles = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(
    f"DrugGEN BACE1 SMILES entries: "
    f"{len(druggen_smiles):,}"
)

druggen_canonical = [
    canonicalize(smiles)
    for smiles in druggen_smiles
]

# ------------------------------------------------------------
# Verify canonicalization
# ------------------------------------------------------------

missing = []

for smiles in druggen_canonical:

    if smiles not in molecule_activity:
        missing.append(smiles)

print()
print("-" * 70)
print("DATA ALIGNMENT")
print("-" * 70)

print(
    f"DrugGEN molecules: {len(druggen_canonical):,}"
)

print(
    f"Missing pIC50 values: {len(missing):,}"
)

if missing:
    print()
    print("ERROR: Some DrugGEN molecules have no pIC50.")

    for smiles in missing[:10]:
        print(smiles)

    raise RuntimeError(
        "Affinity/activity alignment failed."
    )

# ------------------------------------------------------------
# Build molecule-level table
# ------------------------------------------------------------

audit = pd.DataFrame({
    "canonical_smiles": druggen_canonical
})

audit["pIC50"] = audit[
    "canonical_smiles"
].map(molecule_activity)

if audit["pIC50"].isna().any():
    raise RuntimeError(
        "NaN pIC50 values remain after alignment."
    )

# ------------------------------------------------------------
# Calculate reference distribution
# ------------------------------------------------------------

median = audit["pIC50"].median()
p90 = audit["pIC50"].quantile(0.90)

print()
print("-" * 70)
print("REFERENCE VALUES")
print("-" * 70)

print(f"Median pIC50 : {median:.4f}")
print(f"90th pct     : {p90:.4f}")

# ------------------------------------------------------------
# Mild bounded weighting
# ------------------------------------------------------------

def calculate_weight(pic50):

    if pic50 <= median:
        return 1.0

    if pic50 >= p90:
        return 2.0

    return 1.0 + (
        (pic50 - median)
        / (p90 - median)
    )


audit["sampling_weight"] = audit[
    "pIC50"
].apply(calculate_weight)

# ------------------------------------------------------------
# Verify weight range
# ------------------------------------------------------------

print()
print("-" * 70)
print("WEIGHT CHECK")
print("-" * 70)

print(
    f"Minimum weight: "
    f"{audit.sampling_weight.min():.4f}"
)

print(
    f"Maximum weight: "
    f"{audit.sampling_weight.max():.4f}"
)

print(
    f"Mean weight: "
    f"{audit.sampling_weight.mean():.4f}"
)

print(
    f"Weighted / unweighted mean ratio: "
    f"{audit.sampling_weight.mean():.4f}"
)

# ------------------------------------------------------------
# Activity bands
# ------------------------------------------------------------

bins = [
    -np.inf,
    5,
    6,
    7,
    7.6576,
    8,
    9,
    np.inf,
]

labels = [
    "<5",
    "5-6",
    "6-7",
    "7-7.6576",
    "7.6576-8",
    "8-9",
    ">=9",
]

audit["activity_band"] = pd.cut(
    audit["pIC50"],
    bins=bins,
    labels=labels,
    right=False
)

# ------------------------------------------------------------
# Sampling probability by band
# ------------------------------------------------------------

audit["sampling_probability"] = (
    audit["sampling_weight"]
    / audit["sampling_weight"].sum()
)

print()
print("-" * 70)
print("EFFECT OF WEIGHTING BY ACTIVITY BAND")
print("-" * 70)

total_molecules = len(audit)
total_weight = audit["sampling_weight"].sum()

for band in labels:

    subset = audit[
        audit["activity_band"] == band
    ]

    molecules = len(subset)

    if molecules == 0:
        continue

    original_pct = (
        100.0
        * molecules
        / total_molecules
    )

    weighted_probability = (
        subset["sampling_weight"].sum()
        / total_weight
    )

    weighted_pct = 100.0 * weighted_probability

    mean_weight = subset[
        "sampling_weight"
    ].mean()

    print(
        f"{str(band):>12}: "
        f"{molecules:5d} molecules | "
        f"original {original_pct:6.2f}% | "
        f"weighted {weighted_pct:6.2f}% | "
        f"mean weight {mean_weight:.3f}"
    )

# ------------------------------------------------------------
# Important high-affinity checks
# ------------------------------------------------------------

print()
print("-" * 70)
print("HIGH-AFFINITY REGION")
print("-" * 70)

for threshold in [
    7.6576,
    8.0,
    9.0,
]:

    subset = audit[
        audit["pIC50"] >= threshold
    ]

    original_pct = (
        100.0
        * len(subset)
        / total_molecules
    )

    weighted_pct = (
        100.0
        * subset["sampling_weight"].sum()
        / total_weight
    )

    print(
        f"pIC50 >= {threshold:.4f}: "
        f"{len(subset):5d} molecules | "
        f"original {original_pct:6.2f}% | "
        f"weighted {weighted_pct:6.2f}%"
    )

# ------------------------------------------------------------
# Save verification table
# ------------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

audit.to_csv(
    OUTPUT,
    index=False
)

print()
print("-" * 70)
print(f"Saved: {OUTPUT}")
print("-" * 70)

print()
print("VERIFICATION COMPLETE")