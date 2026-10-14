from email import parser
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse

# ============================================================
# Configuration
# ============================================================

INPUT_CSV = Path("results/generated_1024_logBB0.csv")

BACE1_MODEL = Path(
    "results/bace1_activity_model/bace1_activity_model.joblib"
)

BBB_MODEL = Path(
    "results/bbb_activity_model/bbb_model.joblib"
)

OUTPUT_CSV = Path(
    "results/generated_1024_joint_predictions.csv"
)

RADIUS = 2
N_BITS = 2048

B3DB_LOGBB_MEAN = -0.077873
B3DB_LOGBB_STD = 0.751623


# ============================================================
# Molecular representation
# ============================================================

def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=RADIUS,
        nBits=N_BITS
    )

    return np.asarray(fp, dtype=np.float32)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("JOINT BACE1 + BBB PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load generated molecules
    # --------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="results/generated_1024_logBB0.csv",
    )

    parser.add_argument(
        "--output",
        default="results/generated_1024_joint_predictions.csv",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print(f"Input generated records: {len(df)}")

    if "smiles" not in df.columns:
        raise ValueError(
            "Input CSV does not contain a 'smiles' column."
        )

    # Canonicalize and remove invalid structures
    df["canonical_smiles"] = df["smiles"].map(canonicalize)

    invalid = df["canonical_smiles"].isna().sum()

    if invalid:
        print(f"Invalid generated molecules removed: {invalid}")

    df = df.dropna(
        subset=["canonical_smiles"]
    ).copy()

    # Keep one prediction per unique generated molecule
    df = df.drop_duplicates(
        subset=["canonical_smiles"]
    ).reset_index(drop=True)

    print(f"Unique valid molecules: {len(df)}")

    # --------------------------------------------------------
    # Generate fingerprints
    # --------------------------------------------------------

    print()
    print("Generating Morgan fingerprints...")

    X_list = []
    keep = []

    for i, smiles in enumerate(df["canonical_smiles"]):

        fp = fingerprint(smiles)

        if fp is not None:
            X_list.append(fp)
            keep.append(i)

    df = df.iloc[keep].reset_index(drop=True)

    X = np.vstack(X_list).astype(np.float32)

    print(f"Fingerprint shape: {X.shape}")

    # --------------------------------------------------------
    # Load independent models
    # --------------------------------------------------------

    print()
    print("Loading BACE1 activity model...")
    bace1_model = joblib.load(BACE1_MODEL)

    print("Loading BBB regression model...")
    bbb_model = joblib.load(BBB_MODEL)

    # --------------------------------------------------------
    # Predict BACE1 activity
    # --------------------------------------------------------

    print()
    print("Predicting BACE1 activity...")

    predicted_pIC50 = bace1_model.predict(X)

    # --------------------------------------------------------
    # Predict BBB permeability
    # --------------------------------------------------------

    print("Predicting BBB permeability...")

    predicted_logBB = bbb_model.predict(X)

    # --------------------------------------------------------
    # Normalize BBB prediction
    # --------------------------------------------------------

    normalized_logBB = (
        predicted_logBB - B3DB_LOGBB_MEAN
    ) / B3DB_LOGBB_STD

    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    df["predicted_pIC50"] = predicted_pIC50
    df["predicted_logBB"] = predicted_logBB
    df["normalized_predicted_logBB"] = normalized_logBB

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        args.output,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("JOINT PREDICTION SUMMARY")
    print("=" * 70)

    print(
        f"Predicted molecules:       {len(df)}"
    )

    print()
    print("BACE1 predicted pIC50")
    print("-" * 70)
    print(
        f"Mean:   {df['predicted_pIC50'].mean():.4f}"
    )
    print(
        f"Median: {df['predicted_pIC50'].median():.4f}"
    )
    print(
        f"Min:    {df['predicted_pIC50'].min():.4f}"
    )
    print(
        f"Max:    {df['predicted_pIC50'].max():.4f}"
    )

    print()
    print("BBB predicted logBB")
    print("-" * 70)
    print(
        f"Mean:   {df['predicted_logBB'].mean():.4f}"
    )
    print(
        f"Median: {df['predicted_logBB'].median():.4f}"
    )
    print(
        f"Min:    {df['predicted_logBB'].min():.4f}"
    )
    print(
        f"Max:    {df['predicted_logBB'].max():.4f}"
    )

    print()
    print(
        f"Output: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
