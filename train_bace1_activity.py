import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RADIUS = 2
N_BITS = 2048
RANDOM_STATE = 42


def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False
    )


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


def scaffold_split(df):
    groups = {}

    for i, scaf in enumerate(df["scaffold"]):
        groups.setdefault(scaf, []).append(i)

    rng = np.random.RandomState(RANDOM_STATE)
    scaffold_ids = list(groups.keys())
    rng.shuffle(scaffold_ids)

    train_target = int(0.70 * len(df))
    val_target = int(0.15 * len(df))

    train_idx = []
    val_idx = []
    test_idx = []

    for scaf in scaffold_ids:
        members = groups[scaf]

        if len(train_idx) < train_target:
            train_idx.extend(members)
        elif len(val_idx) < val_target:
            val_idx.extend(members)
        else:
            test_idx.extend(members)

    return (
        np.array(train_idx, dtype=int),
        np.array(val_idx, dtype=int),
        np.array(test_idx, dtype=int)
    )


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2": float(r2_score(y_true, y_pred))
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="results/bace1_activity_model"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("BACE1 pIC50 PREDICTOR")
    print("=" * 70)

    df = pd.read_csv(args.input)

    required = {
        "canonical_smiles",
        "pIC50"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df[
        ["canonical_smiles", "pIC50"]
    ].copy()

    df["canonical_smiles"] = (
        df["canonical_smiles"]
        .map(canonicalize)
    )

    df["pIC50"] = pd.to_numeric(
        df["pIC50"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "canonical_smiles",
            "pIC50"
        ]
    )

    # One target per molecule.
    # Multiple measurements are represented by median pIC50.
    df = (
        df.groupby(
            "canonical_smiles",
            as_index=False
        )["pIC50"]
        .median()
    )

    df["scaffold"] = (
        df["canonical_smiles"]
        .map(scaffold)
    )

    df = df.dropna(
        subset=["scaffold"]
    ).reset_index(drop=True)

    fingerprints = []
    keep = []

    print(
        f"Preparing fingerprints for "
        f"{len(df)} unique BACE1 molecules..."
    )

    for i, smi in enumerate(
        df["canonical_smiles"]
    ):

        fp = fingerprint(smi)

        if fp is not None:
            fingerprints.append(fp)
            keep.append(i)

    df = df.iloc[keep].reset_index(drop=True)

    X = np.vstack(
        fingerprints
    ).astype(np.float32)

    y = df["pIC50"].to_numpy(
        dtype=np.float32
    )

    print(
        f"Unique molecules: {len(df)}"
    )

    print(
        f"Unique scaffolds: "
        f"{df['scaffold'].nunique()}"
    )

    print(
        f"Fingerprint shape: {X.shape}"
    )

    # Scaffold split
    train_idx, val_idx, test_idx = (
        scaffold_split(df)
    )

    train_scaffolds = set(
        df.iloc[train_idx]["scaffold"]
    )

    val_scaffolds = set(
        df.iloc[val_idx]["scaffold"]
    )

    test_scaffolds = set(
        df.iloc[test_idx]["scaffold"]
    )

    print()
    print("SCAFFOLD SPLIT")
    print("-" * 70)

    print(
        f"Train:      {len(train_idx)} molecules, "
        f"{len(train_scaffolds)} scaffolds"
    )

    print(
        f"Validation: {len(val_idx)} molecules, "
        f"{len(val_scaffolds)} scaffolds"
    )

    print(
        f"Test:       {len(test_idx)} molecules, "
        f"{len(test_scaffolds)} scaffolds"
    )

    print()
    print("Scaffold overlaps:")

    print(
        "Train ∩ Validation:",
        len(train_scaffolds & val_scaffolds)
    )

    print(
        "Train ∩ Test:",
        len(train_scaffolds & test_scaffolds)
    )

    print(
        "Validation ∩ Test:",
        len(val_scaffolds & test_scaffolds)
    )

    models = {

        "RandomForest":
        RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "ExtraTrees":
        ExtraTreesRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    validation_results = {}
    fitted_models = {}

    print()
    print("VALIDATION")
    print("-" * 70)

    for name, model in models.items():

        print(
            f"Training {name}..."
        )

        model.fit(
            X[train_idx],
            y[train_idx]
        )

        prediction = model.predict(
            X[val_idx]
        )

        result = calculate_metrics(
            y[val_idx],
            prediction
        )

        validation_results[name] = result
        fitted_models[name] = model

        print(
            f"{name}: "
            f"MAE={result['MAE']:.4f}, "
            f"RMSE={result['RMSE']:.4f}, "
            f"R2={result['R2']:.4f}"
        )

    # Select only using validation data.
    selected_name = min(
        validation_results,
        key=lambda name: (
            validation_results[name]["MAE"],
            validation_results[name]["RMSE"]
        )
    )

    selected_model = fitted_models[
        selected_name
    ]

    print()
    print(
        f"Selected model: {selected_name}"
    )

    # Final untouched test.
    test_prediction = selected_model.predict(
        X[test_idx]
    )

    test_result = calculate_metrics(
        y[test_idx],
        test_prediction
    )

    print()
    print("UNTOUCHED SCAFFOLD TEST")
    print("-" * 70)

    print(
        f"MAE:  {test_result['MAE']:.4f}"
    )

    print(
        f"RMSE: {test_result['RMSE']:.4f}"
    )

    print(
        f"R2:   {test_result['R2']:.4f}"
    )

    # Save model.
    model_path = (
        output_dir /
        "bace1_activity_model.joblib"
    )

    joblib.dump(
        selected_model,
        model_path
    )

    metadata = {

        "target": "pIC50",

        "pIC50_definition":
        "9 - log10(IC50[nM])",

        "fingerprint": {
            "type": "Morgan",
            "radius": RADIUS,
            "n_bits": N_BITS
        },

        "random_state":
        RANDOM_STATE,

        "unique_molecules":
        len(df),

        "unique_scaffolds":
        int(df["scaffold"].nunique()),

        "train_samples":
        len(train_idx),

        "validation_samples":
        len(val_idx),

        "test_samples":
        len(test_idx),

        "train_scaffolds":
        len(train_scaffolds),

        "validation_scaffolds":
        len(val_scaffolds),

        "test_scaffolds":
        len(test_scaffolds),

        "validation_results":
        validation_results,

        "selected_model":
        selected_name,

        "test_results":
        test_result
    }

    with open(
        output_dir /
        "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    df.to_csv(
        output_dir /
        "bace1_activity_training_table.csv",
        index=False
    )

    print()
    print("=" * 70)
    print("BACE1 MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Saved model: {model_path}"
    )

    print(
        f"Saved metadata: "
        f"{output_dir / 'model_metadata.json'}"
    )


if __name__ == "__main__":
    main()
