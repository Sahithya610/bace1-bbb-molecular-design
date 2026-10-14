
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Fixed project reference data
# ------------------------------------------------------------

BACE1_DATA = Path("data/bace1_clean.csv")

# Default final-generation input
DEFAULT_GENERATED = Path(
    "results/final_explicit_affinity_bbb_generated_1024.csv"
)

# Default evaluation output
DEFAULT_OUTPUT = Path(
    "results/final_explicit_affinity_bbb_evaluation.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated molecules for BACE1 activity "
            "and continuous BBB permeability."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_GENERATED,
        help=(
            "Generated molecule CSV to evaluate. "
            "Default: results/final_explicit_affinity_bbb_generated_1024.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output evaluation CSV. "
            "Default: results/final_explicit_affinity_bbb_evaluation.csv"
        ),
    )

    return parser.parse_args()


def load_bace1_reference():
    """
    Load experimental BACE1 data and calculate one median
    experimental pIC50 per canonical molecule.
    """

    bace1 = pd.read_csv(BACE1_DATA)

    if "pIC50" not in bace1.columns:

        if "standard_value" in bace1.columns:

            values = pd.to_numeric(
                bace1["standard_value"],
                errors="coerce"
            )

            bace1["pIC50"] = (
                9.0 - np.log10(values)
            )

        else:
            raise ValueError(
                "Could not find pIC50 or standard_value "
                "in BACE1 reference data."
            )

    bace1["pIC50"] = pd.to_numeric(
        bace1["pIC50"],
        errors="coerce"
    )

    bace1 = bace1.dropna(
        subset=["pIC50"]
    ).copy()

    if "canonical_smiles" in bace1.columns:

        bace1_ref = (
            bace1.groupby(
                "canonical_smiles",
                as_index=False
            )["pIC50"]
            .median()
        )

    else:
        bace1_ref = bace1[["pIC50"]].copy()

    return bace1_ref


def load_generated_data(path):
    """
    Load generated molecules and validate the required
    prediction columns.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Generated molecule file not found:\n{path}"
        )

    generated = pd.read_csv(path)

    required_columns = [
        "predicted_pIC50",
        "predicted_logBB",
    ]

    missing = [
        column
        for column in required_columns
        if column not in generated.columns
    ]

    if missing:
        raise ValueError(
            "Generated CSV is missing required columns: "
            + ", ".join(missing)
        )

    return generated


def deduplicate_generated(generated):
    """
    Keep one record per unique molecule when a SMILES column
    is available.

    The generation script already reports valid molecules and
    uniqueness, but this function makes the evaluation robust.
    """

    smiles_column = None

    for column in [
        "canonical_smiles",
        "smiles",
        "SMILES",
        "generated_smiles",
    ]:
        if column in generated.columns:
            smiles_column = column
            break

    if smiles_column is not None:

        before = len(generated)

        generated = generated.dropna(
            subset=[smiles_column]
        ).copy()

        generated = generated.drop_duplicates(
            subset=[smiles_column],
            keep="first"
        ).copy()

        after = len(generated)

        print(
            f"Unique molecules after SMILES deduplication: "
            f"{after}"
        )

        print(
            f"Duplicate generated records removed: "
            f"{before - after}"
        )

    else:

        print(
            "No SMILES column detected; "
            "using all generated prediction records."
        )

    return generated


def main():

    args = parse_args()

    generated_path = args.input
    output_path = args.output

    print("=" * 70)
    print("FINAL BACE1 + BBB GENERATION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load generated predictions
    # --------------------------------------------------------

    print()
    print("GENERATED DATA")
    print("-" * 70)

    print(
        f"Input: {generated_path}"
    )

    generated = load_generated_data(
        generated_path
    )

    print(
        f"Generated records: {len(generated)}"
    )

    # --------------------------------------------------------
    # Deduplicate generated molecules
    # --------------------------------------------------------

    generated = deduplicate_generated(
        generated
    )

    # --------------------------------------------------------
    # Convert predictions to numeric
    # --------------------------------------------------------

    generated["predicted_pIC50"] = pd.to_numeric(
        generated["predicted_pIC50"],
        errors="coerce"
    )

    generated["predicted_logBB"] = pd.to_numeric(
        generated["predicted_logBB"],
        errors="coerce"
    )

    generated = generated.dropna(
        subset=[
            "predicted_pIC50",
            "predicted_logBB",
        ]
    ).copy()

    # --------------------------------------------------------
    # Load experimental BACE1 reference data
    # --------------------------------------------------------

    bace1_ref = load_bace1_reference()

    # --------------------------------------------------------
    # BACE1 reference threshold
    # --------------------------------------------------------

    bace1_threshold = float(
        bace1_ref["pIC50"].quantile(0.75)
    )

    # --------------------------------------------------------
    # BBB threshold
    # --------------------------------------------------------

    bbb_threshold = 0.0

    # --------------------------------------------------------
    # Apply evaluation criteria
    # --------------------------------------------------------

    generated["bace1_active_75pct"] = (
        generated["predicted_pIC50"]
        >= bace1_threshold
    )

    generated["bbb_logBB_ge_0"] = (
        generated["predicted_logBB"]
        >= bbb_threshold
    )

    generated["joint_success"] = (
        generated["bace1_active_75pct"]
        &
        generated["bbb_logBB_ge_0"]
    )

    # --------------------------------------------------------
    # Save molecule-level evaluation
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    generated.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Summary counts
    # --------------------------------------------------------

    total = len(generated)

    bace1_count = int(
        generated["bace1_active_75pct"].sum()
    )

    bbb_count = int(
        generated["bbb_logBB_ge_0"].sum()
    )

    joint_count = int(
        generated["joint_success"].sum()
    )

    # --------------------------------------------------------
    # Reference distribution
    # --------------------------------------------------------

    print()
    print("REFERENCE BACE1 DISTRIBUTION")
    print("-" * 70)

    print(
        f"Reference molecules: "
        f"{len(bace1_ref)}"
    )

    print(
        f"Reference pIC50 median: "
        f"{bace1_ref['pIC50'].median():.4f}"
    )

    print(
        f"Reference pIC50 75th percentile: "
        f"{bace1_threshold:.4f}"
    )

    print(
        f"Reference pIC50 maximum: "
        f"{bace1_ref['pIC50'].max():.4f}"
    )

    # --------------------------------------------------------
    # Generated molecule evaluation
    # --------------------------------------------------------

    print()
    print("GENERATED MOLECULE EVALUATION")
    print("-" * 70)

    print(
        f"Generated unique molecules: "
        f"{total}"
    )

    print(
        f"BACE1 criterion "
        f"(predicted pIC50 >= {bace1_threshold:.4f}): "
        f"{bace1_count} "
        f"({bace1_count / total:.4%})"
    )

    print(
        f"BBB criterion "
        f"(predicted logBB >= {bbb_threshold:.4f}): "
        f"{bbb_count} "
        f"({bbb_count / total:.4%})"
    )

    print(
        f"JOINT criterion: "
        f"{joint_count} "
        f"({joint_count / total:.4%})"
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print()
    print("GENERATED PREDICTION DISTRIBUTION")
    print("-" * 70)

    print(
        f"Mean predicted pIC50: "
        f"{generated['predicted_pIC50'].mean():.4f}"
    )

    print(
        f"Median predicted pIC50: "
        f"{generated['predicted_pIC50'].median():.4f}"
    )

    print(
        f"Maximum predicted pIC50: "
        f"{generated['predicted_pIC50'].max():.4f}"
    )

    print(
        f"Mean predicted logBB: "
        f"{generated['predicted_logBB'].mean():.4f}"
    )

    print(
        f"Median predicted logBB: "
        f"{generated['predicted_logBB'].median():.4f}"
    )

    print(
        f"Maximum predicted logBB: "
        f"{generated['predicted_logBB'].max():.4f}"
    )

    # --------------------------------------------------------
    # Joint candidate property summary
    # --------------------------------------------------------

    print()
    print("JOINT-CANDIDATE PROPERTY SUMMARY")
    print("-" * 70)

    candidates = generated[
        generated["joint_success"]
    ]

    if len(candidates) > 0:

        print(
            f"Candidate count: "
            f"{len(candidates)}"
        )

        print(
            f"Mean predicted pIC50: "
            f"{candidates['predicted_pIC50'].mean():.4f}"
        )

        print(
            f"Median predicted pIC50: "
            f"{candidates['predicted_pIC50'].median():.4f}"
        )

        print(
            f"Maximum predicted pIC50: "
            f"{candidates['predicted_pIC50'].max():.4f}"
        )

        print(
            f"Mean predicted logBB: "
            f"{candidates['predicted_logBB'].mean():.4f}"
        )

        print(
            f"Median predicted logBB: "
            f"{candidates['predicted_logBB'].median():.4f}"
        )

        print(
            f"Maximum predicted logBB: "
            f"{candidates['predicted_logBB'].max():.4f}"
        )

    else:

        print(
            "No molecules satisfy both criteria."
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()

