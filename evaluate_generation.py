"""
DrugGEN generated-molecule evaluation
======================================

Purpose:
    Evaluate the 1,024 generated molecules before downstream
    BACE1 activity and BBB evaluation.

Inputs expected in the DrugGEN project:
    results/generated_1024_logBB0.csv
    data/bace1_clean.csv
    data/chembl_train.smi

Metrics:
    - validity (from generation CSV)
    - uniqueness
    - novelty against BACE1 training SMILES
    - novelty against general ChEMBL training SMILES
    - QED
    - Lipinski HBD/HBA/rotatable bonds
    - molecular weight
    - TPSA
    - Murcko scaffold diversity
    - mean pairwise Morgan Tanimoto similarity

Source/provenance:
    Generated molecules: trained conditional DrugGEN checkpoint,
    BACE1 target condition = 1, desired continuous logBB = 0.0.
    BBB normalization: B3DB mean/std used by the project.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator


def read_smiles_file(path):
    smiles = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()

            if not s:
                continue

            # chembl_train.smi is expected to contain one SMILES per line.
            # If there are extra whitespace-separated fields, use the first.
            s = s.split()[0]

            mol = Chem.MolFromSmiles(s)

            if mol is not None:
                smiles.add(Chem.MolToSmiles(mol, canonical=True))

    return smiles


def canonicalize_smiles(values):
    result = []

    for value in values:
        if pd.isna(value):
            result.append(None)
            continue

        mol = Chem.MolFromSmiles(str(value))

        if mol is None:
            result.append(None)
        else:
            result.append(Chem.MolToSmiles(mol, canonical=True))

    return result


def scaffold_key(mol):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generated",
        default="results/generated_1024_logBB0.csv",
    )

    parser.add_argument(
        "--bace1",
        default="data/bace1_clean.csv",
    )

    parser.add_argument(
        "--chembl",
        default="data/chembl_train.smi",
    )

    parser.add_argument(
        "--output",
        default="results/generated_1024_evaluation.csv",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 70)
    print("DrugGEN Generated Molecule Evaluation")
    print("=" * 70)

    print(f"Generated file: {args.generated}")
    print(f"BACE1 reference: {args.bace1}")
    print(f"ChEMBL reference: {args.chembl}")

    # ------------------------------------------------------------
    # Load generated molecules
    # ------------------------------------------------------------
    generated = pd.read_csv(args.generated)

    if "smiles" not in generated.columns:
        raise ValueError(
            "Expected a 'smiles' column in the generated CSV. "
            f"Found: {list(generated.columns)}"
        )

    generated["canonical_smiles"] = canonicalize_smiles(
        generated["smiles"]
    )

    total = len(generated)

    valid = generated[
        generated["canonical_smiles"].notna()
    ].copy()

    valid_unique = valid["canonical_smiles"].nunique()

    print()
    print(f"Generated graphs: {total}")
    print(f"Valid molecules: {len(valid)}")
    print(f"Unique valid molecules: {valid_unique}")

    validity = len(valid) / total if total else 0.0
    uniqueness = (
        valid_unique / len(valid)
        if len(valid)
        else 0.0
    )

    # ------------------------------------------------------------
    # Reference sets
    # ------------------------------------------------------------
    print()
    print("Loading BACE1 reference molecules...")

    bace1 = pd.read_csv(args.bace1)

    if "canonical_smiles" in bace1.columns:
        bace1_smiles = set(
            canonicalize_smiles(
                bace1["canonical_smiles"]
            )
        )
    elif "SMILES" in bace1.columns:
        bace1_smiles = set(
            canonicalize_smiles(
                bace1["SMILES"]
            )
        )
    else:
        raise ValueError(
            "BACE1 reference must contain canonical_smiles or SMILES."
        )

    bace1_smiles.discard(None)

    print(f"BACE1 reference molecules: {len(bace1_smiles)}")

    print("Loading ChEMBL reference molecules...")

    chembl_smiles = read_smiles_file(args.chembl)

    print(f"ChEMBL reference molecules: {len(chembl_smiles)}")

    # ------------------------------------------------------------
    # Novelty
    # ------------------------------------------------------------
    valid["novel_vs_bace1"] = ~valid[
        "canonical_smiles"
    ].isin(bace1_smiles)

    valid["novel_vs_chembl"] = ~valid[
        "canonical_smiles"
    ].isin(chembl_smiles)

    novel_bace1 = int(valid["novel_vs_bace1"].sum())
    novel_chembl = int(valid["novel_vs_chembl"].sum())

    novelty_bace1 = (
        novel_bace1 / len(valid)
        if len(valid)
        else 0.0
    )

    novelty_chembl = (
        novel_chembl / len(valid)
        if len(valid)
        else 0.0
    )

    # ------------------------------------------------------------
    # Molecular properties
    # ------------------------------------------------------------
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )

    property_rows = []
    fingerprints = []
    scaffolds = []

    for smiles in valid["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        fp = fp_generator.GetFingerprint(mol)
        fingerprints.append(fp)

        scaffold = scaffold_key(mol)

        if scaffold:
            scaffolds.append(scaffold)

        property_rows.append(
            {
                "smiles": smiles,
                "MW": Descriptors.MolWt(mol),
                "LogP": Crippen.MolLogP(mol),
                "TPSA": rdMolDescriptors.CalcTPSA(mol),
                "HBD": Lipinski.NumHDonors(mol),
                "HBA": Lipinski.NumHAcceptors(mol),
                "RotatableBonds": Lipinski.NumRotatableBonds(mol),
                "QED": QED.qed(mol),
                "HeavyAtoms": Lipinski.HeavyAtomCount(mol),
            }
        )

    properties = pd.DataFrame(property_rows)

    # ------------------------------------------------------------
    # Scaffold diversity
    # ------------------------------------------------------------
    unique_scaffolds = len(set(scaffolds))

    scaffold_diversity = (
        unique_scaffolds / len(valid)
        if len(valid)
        else 0.0
    )

    # ------------------------------------------------------------
    # Mean pairwise Morgan Tanimoto similarity
    # ------------------------------------------------------------
    # Sample at most 500 molecules for pairwise similarity so the
    # calculation remains practical while remaining reproducible.
    max_pairwise = min(500, len(fingerprints))

    if max_pairwise > 1:
        selected = np.random.choice(
            len(fingerprints),
            size=max_pairwise,
            replace=False,
        )

        selected_fps = [
            fingerprints[int(i)]
            for i in selected
        ]

        similarities = []

        for i in range(len(selected_fps)):
            sims = DataStructs.BulkTanimotoSimilarity(
                selected_fps[i],
                selected_fps[i + 1:],
            )
            similarities.extend(sims)

        mean_tanimoto = (
            float(np.mean(similarities))
            if similarities
            else 0.0
        )
    else:
        mean_tanimoto = 0.0

    # ------------------------------------------------------------
    # Save per-molecule evaluation
    # ------------------------------------------------------------
    valid = valid.merge(
        properties,
        left_on="canonical_smiles",
        right_on="smiles",
        how="left",
        suffixes=("", "_property"),
    )

    valid = valid.drop(
        columns=["smiles_property"],
        errors="ignore",
    )

    # Add scaffold labels
    scaffold_map = {}

    for smiles in valid["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            scaffold_map[smiles] = scaffold_key(mol)

    valid["murcko_scaffold"] = valid[
        "canonical_smiles"
    ].map(scaffold_map)

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid.to_csv(
        output_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Generated graphs:              {total}")
    print(f"Valid molecules:               {len(valid)}")
    print(f"Validity:                      {validity:.4f}")
    print(f"Unique valid molecules:        {valid_unique}")
    print(f"Uniqueness:                    {uniqueness:.4f}")

    print()
    print(f"BACE1 reference molecules:     {len(bace1_smiles)}")
    print(f"Novel vs BACE1:                {novel_bace1}")
    print(f"Novelty vs BACE1:              {novelty_bace1:.4f}")

    print(f"ChEMBL reference molecules:    {len(chembl_smiles)}")
    print(f"Novel vs ChEMBL:               {novel_chembl}")
    print(f"Novelty vs ChEMBL:             {novelty_chembl:.4f}")

    print()
    print(f"Unique Murcko scaffolds:       {unique_scaffolds}")
    print(f"Scaffold diversity:             {scaffold_diversity:.4f}")
    print(f"Mean pairwise Morgan Tanimoto:  {mean_tanimoto:.4f}")

    if not properties.empty:
        print()
        print("Property means:")
        print(f"  QED:                         {properties['QED'].mean():.4f}")
        print(f"  MW:                          {properties['MW'].mean():.2f}")
        print(f"  LogP:                        {properties['LogP'].mean():.2f}")
        print(f"  TPSA:                        {properties['TPSA'].mean():.2f}")
        print(f"  HBD:                         {properties['HBD'].mean():.2f}")
        print(f"  HBA:                         {properties['HBA'].mean():.2f}")
        print(
            f"  Rotatable bonds:             "
            f"{properties['RotatableBonds'].mean():.2f}"
        )

    print()
    print(f"Detailed output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
