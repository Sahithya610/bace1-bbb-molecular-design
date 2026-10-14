import argparse
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor
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

    return Chem.MolToSmiles(
        mol,
        canonical=True
    )


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False
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

    return np.asarray(
        fp,
        dtype=np.float32
    )


def scaffold_split(df):
    groups = {}

    for i, scaf in enumerate(
        df["scaffold"]
    ):
        groups.setdefault(
            scaf,
            []
        ).append(i)

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    scaffolds = list(
        groups.keys()
    )

    rng.shuffle(
        scaffolds
    )

    train_target = int(
        0.70 * len(df)
    )

    val_target = int(
        0.15 * len(df)
    )

    train = []
    validation = []
    test = []

    for scaf in scaffolds:

        members = groups[scaf]

        if len(train) < train_target:

            train.extend(
                members
            )

        elif len(validation) < val_target:

            validation.extend(
                members
            )

        else:

            test.extend(
                members
            )

    return (
        np.array(
            train,
            dtype=int
        ),
        np.array(
            validation,
            dtype=int
        ),
        np.array(
            test,
            dtype=int
        )
    )


def metrics(y_true, y_pred):

    return {
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            )
        ),
        "R2": float(
            r2_score(
                y_true,
                y_pred
            )
        )
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="B3DB/B3DB/B3DB_classification_extended.tsv.gz"
    )

    parser.add_argument(
        "--output_dir",
        default="results/bbb_activity_model"
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("B3DB QUANTITATIVE BBB REGRESSION MODEL")
    print("=" * 70)

    print(
        f"Input: {args.input}"
    )

    # Read compressed or uncompressed TSV.
    df = pd.read_csv(
        args.input,
        sep="\t",
        compression="gzip" if args.input.endswith(".gz") else None
    )

    print(
        f"Original shape: {df.shape}"
    )

    print(
        "Columns:",
        list(df.columns)
    )

    # Locate required columns.
    smiles_col = None

    for col in [
        "SMILES",
        "smiles",
        "canonical_smiles"
    ]:

        if col in df.columns:

            smiles_col = col
            break

    if smiles_col is None:
        raise ValueError(
            "Could not find SMILES column."
        )

    if "logBB" not in df.columns:
        raise ValueError(
            "Could not find quantitative logBB column."
        )

    work = df[
        [smiles_col, "logBB"]
    ].copy()

    work.columns = [
        "SMILES",
        "logBB"
    ]

    work["SMILES"] = (
        work["SMILES"]
        .map(canonicalize)
    )

    work["logBB"] = pd.to_numeric(
        work["logBB"],
        errors="coerce"
    )

    work = work.dropna(
        subset=[
            "SMILES",
            "logBB"
        ]
    )

    # Remove exact duplicate measurements.
    work = work.drop_duplicates(
        subset=[
            "SMILES",
            "logBB"
        ]
    )

    # One value per structure.
    work = (
        work.groupby(
            "SMILES",
            as_index=False
        )["logBB"]
        .median()
    )

    work["scaffold"] = (
        work["SMILES"]
        .map(get_scaffold)
    )

    work = work.dropna(
        subset=["scaffold"]
    ).reset_index(
        drop=True
    )

    print(
        f"Quantitative molecules: {len(work)}"
    )

    print(
        f"Unique scaffolds: "
        f"{work['scaffold'].nunique()}"
    )

    print()
    print("logBB statistics:")
    print(
        work["logBB"].describe()
    )

    # Morgan fingerprints.
    X_list = []
    keep = []

    print()
    print(
        "Generating Morgan fingerprints..."
    )

    for i, smi in enumerate(
        work["SMILES"]
    ):

        fp = fingerprint(
            smi
        )

        if fp is not None:

            X_list.append(
                fp
            )

            keep.append(
                i
            )

    work = work.iloc[
        keep
    ].reset_index(
        drop=True
    )

    X = np.vstack(
        X_list
    ).astype(
        np.float32
    )

    y = work[
        "logBB"
    ].to_numpy(
        dtype=np.float32
    )

    print(
        f"Fingerprint shape: {X.shape}"
    )

    # Scaffold split.
    train_idx, val_idx, test_idx = (
        scaffold_split(work)
    )

    train_scaffolds = set(
        work.iloc[
            train_idx
        ]["scaffold"]
    )

    val_scaffolds = set(
        work.iloc[
            val_idx
        ]["scaffold"]
    )

    test_scaffolds = set(
        work.iloc[
            test_idx
        ]["scaffold"]
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
        len(
            train_scaffolds &
            val_scaffolds
        )
    )

    print(
        "Train ∩ Test:",
        len(
            train_scaffolds &
            test_scaffolds
        )
    )

    print(
        "Validation ∩ Test:",
        len(
            val_scaffolds &
            test_scaffolds
        )
    )

    # Exact established ExtraTrees configuration.
    model = ExtraTreesRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    print()
    print(
        "TRAINING EXTRATREES"
    )
    print("-" * 70)

    model.fit(
        X[train_idx],
        y[train_idx]
    )

    validation_prediction = (
        model.predict(
            X[val_idx]
        )
    )

    validation_result = metrics(
        y[val_idx],
        validation_prediction
    )

    print(
        f"Validation MAE:  "
        f"{validation_result['MAE']:.4f}"
    )

    print(
        f"Validation RMSE: "
        f"{validation_result['RMSE']:.4f}"
    )

    print(
        f"Validation R2:   "
        f"{validation_result['R2']:.4f}"
    )

    test_prediction = (
        model.predict(
            X[test_idx]
        )
    )

    test_result = metrics(
        y[test_idx],
        test_prediction
    )

    print()
    print(
        "UNTOUCHED SCAFFOLD TEST"
    )
    print("-" * 70)

    print(
        f"Test MAE:  "
        f"{test_result['MAE']:.4f}"
    )

    print(
        f"Test RMSE: "
        f"{test_result['RMSE']:.4f}"
    )

    print(
        f"Test R2:   "
        f"{test_result['R2']:.4f}"
    )

    # Save model.
    model_path = (
        output_dir /
        "bbb_model.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    metadata = {

        "target": "logBB",

        "dataset":
        "B3DB quantitative logBB",

        "fingerprint": {
            "type": "Morgan",
            "radius": RADIUS,
            "n_bits": N_BITS
        },

        "random_state":
        RANDOM_STATE,

        "n_estimators":
        500,

        "max_features":
        "sqrt",

        "min_samples_leaf":
        2,

        "molecules":
        len(work),

        "scaffolds":
        int(
            work["scaffold"].nunique()
        ),

        "train_samples":
        len(train_idx),

        "validation_samples":
        len(val_idx),

        "test_samples":
        len(test_idx),

        "validation_results":
        validation_result,

        "test_results":
        test_result,

        "B3DB_logBB_mean":
        -0.077873,

        "B3DB_logBB_std":
        0.751623
    }

    with open(
        output_dir /
        "bbb_model_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("BBB MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Saved model: {model_path}"
    )

    print(
        f"Saved metadata: "
        f"{output_dir / 'bbb_model_metadata.json'}"
    )


if __name__ == "__main__":
    main()
