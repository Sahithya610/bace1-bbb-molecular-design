"""
Precompute ExtraTrees pIC50 predictions for all BACE1 training molecules.

Purpose
-------
The condition-consistency loss in train_joint.py trains the generator to
satisfy D.predict_properties — a property head that reads post-Transformer
graph features.  That head is currently supervised by raw pIC50 labels from
the training CSV.

The external evaluation oracle is an ExtraTrees model on Morgan ECFP4
fingerprints.  The two representations are decoupled: satisfying the internal
head does not move the external oracle score.

Fix: re-supervise the property head with *ExtraTrees predictions* on the
training molecules, rather than raw pIC50 labels.  The head then learns to
predict "what ExtraTrees would say" from graph features — bridging the two
spaces without any differentiable fingerprint approximation or soft FP.

This script:
  1. Loads the trained ExtraTrees model.
  2. Computes Morgan ECFP4 fingerprints for all 8,271 BACE1 training molecules.
  3. Runs ExtraTrees inference to get predicted pIC50 for each molecule.
  4. Writes druggen_data/bace1_et_predictions.csv with columns:
       canonical_smiles, pIC50_raw, et_pred_pIC50, et_pred_norm_pIC50

Usage
-----
    .venv\\Scripts\\python.exe precompute_et_labels.py

Outputs
-------
    druggen_data/bace1_et_predictions.csv
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# ============================================================
# Constants — must match train_bace1_activity.py and predict_joint.py
# ============================================================

RADIUS = 2
N_BITS = 2048

PIC50_MEAN = 6.691315
PIC50_STD = 1.308678

INPUT_CSV = Path("druggen_data/bace1_continuous_bbb.csv")
MODEL_PATH = Path("results/bace1_activity_model/bace1_activity_model.joblib")
OUTPUT_CSV = Path("druggen_data/bace1_et_predictions.csv")


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
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=RADIUS, nBits=N_BITS)
    return np.asarray(fp, dtype=np.float32)


def main():
    print("=" * 70)
    print("PRECOMPUTE ExtraTrees pIC50 PREDICTIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ExtraTrees model not found: {MODEL_PATH}\n"
            "Run train_bace1_activity.py first."
        )

    df = pd.read_csv(INPUT_CSV)
    print(f"Input records: {len(df)}")

    df["canonical_smiles"] = df["canonical_smiles"].map(canonicalize)
    df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    df = df.dropna(subset=["canonical_smiles", "pIC50"])

    # One row per unique molecule (median pIC50, same as training).
    df = df.groupby("canonical_smiles", as_index=False)["pIC50"].median()
    print(f"Unique molecules after dedup: {len(df)}")

    # --------------------------------------------------------
    # Fingerprints
    # --------------------------------------------------------
    fps, keep = [], []
    for i, smi in enumerate(df["canonical_smiles"]):
        fp = fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            keep.append(i)

    df = df.iloc[keep].reset_index(drop=True)
    X = np.vstack(fps).astype(np.float32)
    print(f"Fingerprint matrix: {X.shape}")

    # --------------------------------------------------------
    # ExtraTrees predictions
    # --------------------------------------------------------
    print(f"Loading ExtraTrees from: {MODEL_PATH}")
    et = joblib.load(MODEL_PATH)
    et_preds = et.predict(X).astype(np.float32)   # raw pIC50

    # Normalize to the same space the generator uses.
    et_preds_norm = (et_preds - PIC50_MEAN) / PIC50_STD

    # --------------------------------------------------------
    # Assemble and save
    # --------------------------------------------------------
    out_df = pd.DataFrame({
        "canonical_smiles":  df["canonical_smiles"].values,
        "pIC50_raw":         df["pIC50"].values,
        "et_pred_pIC50":     et_preds,
        "et_pred_norm_pIC50": et_preds_norm,
    })

    # Quick sanity check
    from sklearn.metrics import mean_absolute_error, r2_score
    r2  = r2_score(out_df["pIC50_raw"], out_df["et_pred_pIC50"])
    mae = mean_absolute_error(out_df["pIC50_raw"], out_df["et_pred_pIC50"])
    print(f"\nET self-prediction (train+val+test pooled):")
    print(f"  R²  = {r2:.4f}   (should be ~1.0 for train, <0.7 for test)")
    print(f"  MAE = {mae:.4f}")
    print(f"\nET pred stats:")
    print(f"  mean  = {et_preds.mean():.4f}")
    print(f"  std   = {et_preds.std():.4f}")
    print(f"  min   = {et_preds.min():.4f}")
    print(f"  max   = {et_preds.max():.4f}")

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(out_df)} rows)")

    # Save metadata
    meta = {
        "input_csv":      str(INPUT_CSV),
        "model_path":     str(MODEL_PATH),
        "output_csv":     str(OUTPUT_CSV),
        "n_molecules":    len(out_df),
        "fingerprint":    {"type": "Morgan", "radius": RADIUS, "n_bits": N_BITS},
        "normalization":  {"mean": PIC50_MEAN, "std": PIC50_STD},
        "pooled_r2":      float(r2),
        "pooled_mae":     float(mae),
        "et_pred_mean":   float(et_preds.mean()),
        "et_pred_std":    float(et_preds.std()),
    }
    meta_path = OUTPUT_CSV.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
