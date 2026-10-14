from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders


BACE1_CSV = Path("data/bace1_clean.csv")
RAW_DIR = Path("druggen_data")


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


print("=" * 70)
print("ACTUAL DRUGGEN BACE1 DATASET ALIGNMENT")
print("=" * 70)

# ------------------------------------------------------------
# Load activity data
# ------------------------------------------------------------

df = pd.read_csv(BACE1_CSV)

if "pIC50" not in df.columns:
    df["standard_value"] = pd.to_numeric(
        df["standard_value"],
        errors="coerce"
    )
    df["pIC50"] = 9.0 - np.log10(df["standard_value"])

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

activity = (
    df.groupby("canonical_smiles")["pIC50"]
    .median()
)

print(f"Activity molecules: {len(activity):,}")

# ------------------------------------------------------------
# Load DrugGEN encoders
# ------------------------------------------------------------

raw_file = str(RAW_DIR / "bace1.smi")
drug_raw_file = str(Path("data") / "chembl_train.smi")

print()
print("Loading DrugGEN encoders/decoders...")

atom_encoder, atom_decoder, bond_encoder, bond_decoder = (
    get_encoders_decoders(
        raw_file,
        drug_raw_file,
        45
    )
)

# ------------------------------------------------------------
# Load DrugGEN dataset using the SAME class as training
# ------------------------------------------------------------

print()
print("Loading DrugGEN BACE1 dataset...")

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

print()
print("-" * 70)
print("ACTUAL DATASET")
print("-" * 70)

print(f"DrugGEN graph count: {len(dataset):,}")

# ------------------------------------------------------------
# IMPORTANT:
# DruggenDataset contains the processed graph representation.
#
# We cannot safely infer the original SMILES simply from
# graph object length. Instead, verify that the dataset's
# processed count matches the known preprocessing result.
# ------------------------------------------------------------

if len(dataset) != 7682:
    raise RuntimeError(
        f"Expected 7,682 DrugGEN BACE1 graphs, "
        f"but loaded {len(dataset):,}."
    )

print("Graph count matches expected 7,682.")

# ------------------------------------------------------------
# Inspect graph objects
# ------------------------------------------------------------

print()
print("-" * 70)
print("GRAPH STRUCTURE")
print("-" * 70)

sample = dataset[0]

print(f"First graph type: {type(sample)}")

if hasattr(sample, "x"):
    print(f"Node tensor shape: {sample.x.shape}")

if hasattr(sample, "edge_index"):
    print(
        f"Edge index shape: "
        f"{sample.edge_index.shape}"
    )

if hasattr(sample, "edge_attr"):
    print(
        f"Edge attribute shape: "
        f"{sample.edge_attr.shape}"
    )

# ------------------------------------------------------------
# IMPORTANT STOP
# ------------------------------------------------------------

print()
print("=" * 70)
print("DATASET COUNT VERIFIED")
print("=" * 70)

print()
print(
    "The DrugGEN dataset contains exactly 7,682 graphs."
)

print(
    "We will NOT assume that simple SMILES filtering reproduces "
    "the DrugGEN preprocessing."
)

print()
print(
    "Next step: inspect how DruggenDataset preserves molecule "
    "identity so the pIC50 weights can be mapped safely."
)