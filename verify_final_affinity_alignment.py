"""
Final verification of DrugGEN BACE1 graph -> pIC50 -> weight mapping.

Uses the actual DruggenDataset objects and their stored SMILES.

NO dataset modification.
NO model modification.
NO training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders


CSV_PATH = Path("data/bace1_clean.csv")
RAW_DIR = Path("druggen_data")
OUTPUT = Path("results/bace1_final_affinity_alignment.csv")


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


print("=" * 70)
print("FINAL BACE1 GRAPH -> pIC50 ALIGNMENT")
print("=" * 70)

# ------------------------------------------------------------
# Load BACE1 activity data
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

print(f"Activity molecules: {len(activity):,}")

# ------------------------------------------------------------
# Load DrugGEN encoders
# ------------------------------------------------------------

print()
print("Loading DrugGEN encoders/decoders...")

atom_encoder, atom_decoder, bond_encoder, bond_decoder = (
    get_encoders_decoders(
        str(RAW_DIR / "bace1.smi"),
        "data/chembl_train.smi",
        45
    )
)

# ------------------------------------------------------------
# Load ACTUAL processed DrugGEN dataset
# ------------------------------------------------------------

print()
print("Loading actual DrugGEN BACE1 dataset...")

dataset = DruggenDataset(
    root=str(RAW_DIR),
    dataset_file="bace145.pt",
    raw_files=["bace1.smi"],
    max_atom=45,
    features=False,
    atom_encoder=atom_encoder,
    atom_decoder=atom_decoder,
    bond_encoder=bond_encoder,
    bond_decoder=bond_decoder,
)

print(f"DrugGEN graphs: {len(dataset):,}")

if len(dataset) != 7682:
    raise RuntimeError(
        f"Expected 7,682 graphs, got {len(dataset)}"
    )

# ------------------------------------------------------------
# Extract SMILES DIRECTLY FROM ACTUAL GRAPH OBJECTS
# ------------------------------------------------------------

records = []

for index in range(len(dataset)):

    graph = dataset[index]

    if not hasattr(graph, "smiles"):
        raise RuntimeError(
            f"Graph {index} has no smiles attribute."
        )

    raw_smiles = graph.smiles

    canonical_smiles = canonicalize(raw_smiles)

    if canonical_smiles is None:
        raise RuntimeError(
            f"Graph {index} has invalid SMILES: "
            f"{raw_smiles}"
        )

    records.append({
        "graph_index": index,
        "smiles": raw_smiles,
        "canonical_smiles": canonical_smiles,
    })

aligned = pd.DataFrame(records)

# ------------------------------------------------------------
# Map pIC50
# ------------------------------------------------------------

aligned["pIC50"] = aligned[
    "canonical_smiles"
].map(activity)

missing = aligned["pIC50"].isna().sum()

print()
print("-" * 70)
print("GRAPH -> MOLECULE -> pIC50")
print("-" * 70)

print(f"Graphs checked:        {len(aligned):,}")
print(f"pIC50 values matched:  {aligned.pIC50.notna().sum():,}")
print(f"Missing pIC50:         {missing:,}")

if missing > 0:

    print()
    print("First missing molecules:")

    print(
        aligned[
            aligned["pIC50"].isna()
        ][["graph_index", "smiles"]]
        .head(10)
        .to_string(index=False)
    )

    raise RuntimeError(
        "Graph-to-pIC50 alignment failed."
    )

# ------------------------------------------------------------
# Calculate weighting thresholds
# ------------------------------------------------------------

median = aligned["pIC50"].median()
p90 = aligned["pIC50"].quantile(0.90)

print()
print("-" * 70)
print("WEIGHTING THRESHOLDS")
print("-" * 70)

print(f"Median pIC50: {median:.4f}")
print(f"90th percentile: {p90:.4f}")

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


aligned["sampling_weight"] = aligned[
    "pIC50"
].apply(calculate_weight)

# ------------------------------------------------------------
# Verify weights
# ------------------------------------------------------------

print()
print("-" * 70)
print("WEIGHT VERIFICATION")
print("-" * 70)

print(
    f"Minimum weight: {aligned.sampling_weight.min():.4f}"
)

print(
    f"Maximum weight: {aligned.sampling_weight.max():.4f}"
)

print(
    f"Mean weight: {aligned.sampling_weight.mean():.4f}"
)

print(
    f"Missing weights: "
    f"{aligned.sampling_weight.isna().sum():,}"
)

# ------------------------------------------------------------
# Activity-band probability
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

aligned["activity_band"] = pd.cut(
    aligned["pIC50"],
    bins=bins,
    labels=labels,
    right=False
)

total_weight = aligned[
    "sampling_weight"
].sum()

print()
print("-" * 70)
print("SAMPLING DISTRIBUTION")
print("-" * 70)

for band in labels:

    subset = aligned[
        aligned["activity_band"] == band
    ]

    molecule_count = len(subset)

    if molecule_count == 0:
        continue

    original_pct = (
        100.0
        * molecule_count
        / len(aligned)
    )

    weighted_pct = (
        100.0
        * subset["sampling_weight"].sum()
        / total_weight
    )

    print(
        f"{str(band):>12}: "
        f"{molecule_count:5d} graphs | "
        f"original {original_pct:6.2f}% | "
        f"weighted {weighted_pct:6.2f}%"
    )

# ------------------------------------------------------------
# Final high-affinity check
# ------------------------------------------------------------

print()
print("-" * 70)
print("HIGH-AFFINITY SAMPLING SHIFT")
print("-" * 70)

for threshold in [7.6576, 8.0, 9.0]:

    subset = aligned[
        aligned["pIC50"] >= threshold
    ]

    original_pct = (
        100.0
        * len(subset)
        / len(aligned)
    )

    weighted_pct = (
        100.0
        * subset["sampling_weight"].sum()
        / total_weight
    )

    print(
        f"pIC50 >= {threshold:.4f}: "
        f"{len(subset):5d} graphs | "
        f"original {original_pct:6.2f}% | "
        f"weighted {weighted_pct:6.2f}%"
    )

# ------------------------------------------------------------
# Save exact graph-index mapping
# ------------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

aligned.to_csv(
    OUTPUT,
    index=False
)

print()
print("-" * 70)
print(f"Saved: {OUTPUT}")
print("-" * 70)

print()
print("=" * 70)
print("FINAL ALIGNMENT VERIFIED")
print("=" * 70)