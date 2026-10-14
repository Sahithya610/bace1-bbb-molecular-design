import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def read_generated(path):
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise ValueError(f"Generated file must contain 'smiles'. Found: {list(df.columns)}")
    return df


def read_bace1(path):
    df = pd.read_csv(path)
    smi_col = "canonical_smiles" if "canonical_smiles" in df.columns else "SMILES"
    if smi_col not in df.columns:
        raise ValueError("BACE1 CSV must contain canonical_smiles or SMILES.")
    return df, smi_col


def read_chembl_smi(path):
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            token = line.strip().split()
            if token:
                out.append(token[0])
    return out


def morgan_fp(mol, radius=2, n_bits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def scaffold_key(mol):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def pairwise_mean_tanimoto(mols, max_mols=1000):
    from rdkit.DataStructs import TanimotoSimilarity
    mols = mols[:max_mols]
    fps = [morgan_fp(m) for m in mols]
    if len(fps) < 2:
        return 0.0
    vals = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            vals.append(TanimotoSimilarity(fps[i], fps[j]))
    return float(np.mean(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--bace1", required=True)
    ap.add_argument("--chembl", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gen = read_generated(args.generated)
    bace, bace_smi_col = read_bace1(args.bace1)
    chembl_smiles = read_chembl_smi(args.chembl)

    # IMPORTANT: exactly one row per generated record.
    generated_rows = len(gen)
    valid_records = []
    invalid_records = 0

    for idx, smi in enumerate(gen["smiles"].tolist()):
        cs = canonicalize(smi)
        if cs is None:
            invalid_records += 1
            continue
        mol = Chem.MolFromSmiles(cs)
        valid_records.append((idx, cs, mol))

    valid_count = len(valid_records)
    unique_valid = len(set(x[1] for x in valid_records))

    gen_unique = {}
    for idx, cs, mol in valid_records:
        gen_unique.setdefault(cs, (idx, mol))

    bace_set = set()
    for smi in bace[bace_smi_col].dropna().astype(str):
        cs = canonicalize(smi)
        if cs:
            bace_set.add(cs)

    chembl_set = set()
    for smi in chembl_smiles:
        cs = canonicalize(smi)
        if cs:
            chembl_set.add(cs)

    unique_generated = [(idx, cs, mol) for cs, (idx, mol) in gen_unique.items()]

    novel_bace = [x for x in unique_generated if x[1] not in bace_set]
    novel_chembl = [x for x in unique_generated if x[1] not in chembl_set]

    mols = [x[2] for x in unique_generated]
    scaffolds = {scaffold_key(m) for m in mols}
    scaffolds.discard("")
    scaffold_diversity = len(scaffolds) / len(mols) if mols else 0.0
    mean_tanimoto = pairwise_mean_tanimoto(mols)

    rows = []
    for idx, cs, mol in valid_records:
        rows.append({
            "generated_row": idx,
            "smiles_original": gen.iloc[idx]["smiles"],
            "canonical_smiles": cs,
            "is_unique": cs in gen_unique and gen_unique[cs][0] == idx,
            "novel_vs_bace1": cs not in bace_set,
            "novel_vs_chembl": cs not in chembl_set,
            "QED": QED.qed(mol),
            "MW": Descriptors.MolWt(mol),
            "LogP": Crippen.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotatableBonds": Lipinski.NumRotatableBonds(mol),
            "MurckoScaffold": scaffold_key(mol),
        })

    detail = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False)

    print("=" * 70)
    print("CORRECTED GENERATION EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Generated records:             {generated_rows}")
    print(f"Valid molecules:               {valid_count}")
    print(f"Invalid molecules:             {invalid_records}")
    print(f"Validity:                      {valid_count / generated_rows if generated_rows else 0:.4f}")
    print(f"Unique valid molecules:        {unique_valid}")
    print(f"Uniqueness:                    {unique_valid / valid_count if valid_count else 0:.4f}")
    print(f"BACE1 reference molecules:     {len(bace_set)}")
    print(f"Novel unique vs BACE1:         {len(novel_bace)}")
    print(f"Novelty vs BACE1:              {len(novel_bace) / unique_valid if unique_valid else 0:.4f}")
    print(f"ChEMBL reference molecules:    {len(chembl_set)}")
    print(f"Novel unique vs ChEMBL:        {len(novel_chembl)}")
    print(f"Novelty vs ChEMBL:              {len(novel_chembl) / unique_valid if unique_valid else 0:.4f}")
    print(f"Unique Murcko scaffolds:        {len(scaffolds)}")
    print(f"Scaffold diversity:             {scaffold_diversity:.4f}")
    print(f"Mean pairwise Morgan Tanimoto:  {mean_tanimoto:.4f}")

    if not detail.empty:
        print("\nProperty means:")
        for col, label in [
            ("QED", "QED"),
            ("MW", "MW"),
            ("LogP", "LogP"),
            ("TPSA", "TPSA"),
            ("HBD", "HBD"),
            ("HBA", "HBA"),
            ("RotatableBonds", "Rotatable bonds"),
        ]:
            print(f"  {label:28s} {detail[col].mean():.4f}")

    print(f"\nDetailed output: {args.output}")


if __name__ == "__main__":
    main()
