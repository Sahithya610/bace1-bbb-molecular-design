"""
Direct prediction script that writes output CSV immediately upon completion.
Suppresses RDKit deprecation warnings, redirects all progress to a status file.
"""
import sys
import os
import warnings

# Suppress all warnings (including RDKit MorganGenerator deprecation)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Force unbuffered output
sys.stdout = open(
    r"c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\results\predict_direct_log.txt",
    "w", buffering=1  # line-buffered
)
sys.stderr = open(
    r"c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\results\predict_direct_err.txt",
    "w", buffering=1
)

import traceback

try:
    print("Starting prediction pipeline...", flush=True)

    import joblib
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    # Suppress RDKit C++ warnings
    RDLogger.DisableLog("rdApp.*")

    INPUT_CSV  = r"c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\results\baseline_screen_50000_generated.csv"
    OUTPUT_CSV = r"c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\results\baseline_screen_50000_predictions.csv"
    BACE1_MODEL = r"c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\results\bace1_activity_model\bace1_activity_model.joblib"
    BBB_MODEL   = r"c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master\results\bbb_activity_model\bbb_model.joblib"

    RADIUS = 2
    N_BITS = 2048
    B3DB_LOGBB_MEAN = -0.077873
    B3DB_LOGBB_STD  =  0.751623

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

    # ---- Load input
    print(f"Loading {INPUT_CSV} ...", flush=True)
    df = pd.read_csv(INPUT_CSV)
    print(f"Input records: {len(df)}", flush=True)

    if "smiles" not in df.columns:
        raise ValueError("Input CSV missing 'smiles' column")

    # ---- Canonicalize
    print("Canonicalizing SMILES...", flush=True)
    df["canonical_smiles"] = df["smiles"].map(canonicalize)
    invalid = df["canonical_smiles"].isna().sum()
    print(f"Invalid (non-parseable) SMILES: {invalid}", flush=True)

    df = df.dropna(subset=["canonical_smiles"]).copy()
    df = df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    print(f"Unique valid molecules: {len(df)}", flush=True)

    # ---- Fingerprints
    print("Computing Morgan fingerprints (radius=2, nBits=2048)...", flush=True)
    X_list = []
    keep = []
    for i, smiles in enumerate(df["canonical_smiles"]):
        fp = fingerprint(smiles)
        if fp is not None:
            X_list.append(fp)
            keep.append(i)
        if (i + 1) % 5000 == 0:
            print(f"  Fingerprinted {i+1}/{len(df)}...", flush=True)

    df = df.iloc[keep].reset_index(drop=True)
    X = np.vstack(X_list).astype(np.float32)
    print(f"Fingerprint matrix shape: {X.shape}", flush=True)

    # ---- Load models
    print("Loading BACE1 model...", flush=True)
    bace1_model = joblib.load(BACE1_MODEL)
    print("Loading BBB model...", flush=True)
    bbb_model = joblib.load(BBB_MODEL)

    # ---- Predict
    print("Predicting BACE1 pIC50...", flush=True)
    predicted_pIC50 = bace1_model.predict(X)
    print("Predicting BBB logBB...", flush=True)
    predicted_logBB = bbb_model.predict(X)
    normalized_logBB = (predicted_logBB - B3DB_LOGBB_MEAN) / B3DB_LOGBB_STD

    df["predicted_pIC50"] = predicted_pIC50
    df["predicted_logBB"] = predicted_logBB
    df["normalized_predicted_logBB"] = normalized_logBB

    # ---- Save
    print(f"Saving to {OUTPUT_CSV} ...", flush=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows.", flush=True)

    # ---- Summary
    print("", flush=True)
    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total predicted: {len(df)}", flush=True)
    print(f"pIC50  mean={df['predicted_pIC50'].mean():.4f}  median={df['predicted_pIC50'].median():.4f}  "
          f"min={df['predicted_pIC50'].min():.4f}  max={df['predicted_pIC50'].max():.4f}", flush=True)
    print(f"logBB  mean={df['predicted_logBB'].mean():.4f}  median={df['predicted_logBB'].median():.4f}  "
          f"min={df['predicted_logBB'].min():.4f}  max={df['predicted_logBB'].max():.4f}", flush=True)

    n_pic50_6   = (df['predicted_pIC50'] >= 6.0).sum()
    n_pic50_7   = (df['predicted_pIC50'] >= 7.0).sum()
    n_pic50_765 = (df['predicted_pIC50'] >= 7.6576).sum()
    n_logBB_0   = (df['predicted_logBB']  >= 0.0).sum()
    n_joint     = ((df['predicted_pIC50'] >= 7.6576) & (df['predicted_logBB'] >= 0.0)).sum()

    print(f"pIC50 >= 6.0:     {n_pic50_6}  ({100*n_pic50_6/len(df):.3f}%)", flush=True)
    print(f"pIC50 >= 7.0:     {n_pic50_7}  ({100*n_pic50_7/len(df):.3f}%)", flush=True)
    print(f"pIC50 >= 7.6576:  {n_pic50_765}  ({100*n_pic50_765/len(df):.3f}%)", flush=True)
    print(f"logBB >= 0.0:     {n_logBB_0}  ({100*n_logBB_0/len(df):.3f}%)", flush=True)
    print(f"JOINT (both):     {n_joint}  ({100*n_joint/len(df):.3f}%)", flush=True)
    print("DONE", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    sys.exit(1)

sys.stdout.flush()
sys.stdout.close()
sys.stderr.close()
