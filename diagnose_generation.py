from pathlib import Path

import numpy as np
import pandas as pd


GENERATED = Path(
    "results/generated_1024_joint_predictions.csv"
)

BACE1_DATA = Path(
    "data/bace1_clean.csv"
)


def describe(name, values):

    values = pd.Series(values).dropna()

    print()
    print(name)
    print("-" * 70)

    print(f"Count:   {len(values)}")
    print(f"Mean:    {values.mean():.4f}")
    print(f"Median:  {values.median():.4f}")
    print(f"Std:     {values.std():.4f}")
    print(f"Min:     {values.min():.4f}")
    print(f"25%:     {values.quantile(0.25):.4f}")
    print(f"75%:     {values.quantile(0.75):.4f}")
    print(f"Max:     {values.max():.4f}")


def fraction_above(values, threshold):

    values = pd.Series(values).dropna()

    count = int(
        (values >= threshold).sum()
    )

    return count, count / len(values)


def main():

    print("=" * 70)
    print("BACE1 GENERATION DIAGNOSTIC")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    generated = pd.read_csv(
        GENERATED
    )

    reference = pd.read_csv(
        BACE1_DATA
    )

    # --------------------------------------------------------
    # Reference pIC50
    # --------------------------------------------------------

    if "pIC50" not in reference.columns:

        reference["standard_value"] = pd.to_numeric(
            reference["standard_value"],
            errors="coerce"
        )

        reference["pIC50"] = (
            9.0
            - np.log10(
                reference["standard_value"]
            )
        )

    reference["pIC50"] = pd.to_numeric(
        reference["pIC50"],
        errors="coerce"
    )

    reference = reference.dropna(
        subset=["pIC50"]
    )

    # Median per molecule
    if "canonical_smiles" in reference.columns:

        reference = (
            reference.groupby(
                "canonical_smiles",
                as_index=False
            )["pIC50"]
            .median()
        )

    generated_p = generated[
        "predicted_pIC50"
    ]

    reference_p = reference[
        "pIC50"
    ]

    # --------------------------------------------------------
    # Distributions
    # --------------------------------------------------------

    describe(
        "EXPERIMENTAL BACE1 REFERENCE",
        reference_p
    )

    describe(
        "GENERATED PREDICTED BACE1",
        generated_p
    )

    # --------------------------------------------------------
    # Threshold comparison
    # --------------------------------------------------------

    print()
    print("THRESHOLD COMPARISON")
    print("-" * 70)

    thresholds = [
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        7.6576
    ]

    print(
        f"{'Threshold':<12}"
        f"{'Reference':<18}"
        f"{'Generated':<18}"
    )

    print("-" * 48)

    for threshold in thresholds:

        ref_count, ref_fraction = fraction_above(
            reference_p,
            threshold
        )

        gen_count, gen_fraction = fraction_above(
            generated_p,
            threshold
        )

        print(
            f"{threshold:<12.4f}"
            f"{ref_count:>5} "
            f"({ref_fraction:>7.2%})   "
            f"{gen_count:>5} "
            f"({gen_fraction:>7.2%})"
        )

    # --------------------------------------------------------
    # BBB distribution
    # --------------------------------------------------------

    describe(
        "GENERATED PREDICTED BBB logBB",
        generated["predicted_logBB"]
    )

    # --------------------------------------------------------
    # Correlation between generated properties
    # --------------------------------------------------------

    correlation = generated[
        [
            "predicted_pIC50",
            "predicted_logBB"
        ]
    ].corr().iloc[0, 1]

    print()
    print("GENERATED PROPERTY CORRELATION")
    print("-" * 70)
    print(
        f"pIC50 vs logBB correlation: "
        f"{correlation:.4f}"
    )

    # --------------------------------------------------------
    # Best generated molecules
    # --------------------------------------------------------

    print()
    print("TOP 20 GENERATED MOLECULES BY PREDICTED pIC50")
    print("-" * 70)

    top = generated.sort_values(
        "predicted_pIC50",
        ascending=False
    ).head(20)

    columns = [
        "canonical_smiles",
        "predicted_pIC50",
        "predicted_logBB"
    ]

    print(
        top[columns].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()