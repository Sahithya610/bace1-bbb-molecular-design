"""
Audit 1C: Oracle Model Verification
Verifies all metadata claims, checks scaffold overlap, duplicate SMILES,
pIC50 definition consistency, and computes test-set metrics if predictions
are available.
"""

import sys
import warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
MODEL_DIR   = ROOT / "results" / "bace1_activity_model"
META_FILE   = MODEL_DIR / "model_metadata.json"
TRAIN_CSV   = MODEL_DIR / "bace1_activity_training_table.csv"
OUT_DIR     = ROOT / "reports" / "audit_corrections"
OUT_JSON    = OUT_DIR / "oracle_metrics_verified.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load and print metadata ─────────────────────────────────────────
print("=" * 70)
print("ORACLE MODEL METADATA")
print("=" * 70)
meta = json.loads(META_FILE.read_text(encoding="utf-8"))
for k, v in meta.items():
    print(f"  {k}: {v}")

# ── 2. Confirm key metrics ─────────────────────────────────────────────
test_res     = meta.get("test_results", {})
reported_n   = meta.get("test_samples", None)
reported_r2  = test_res.get("R2",   None)
reported_rmse= test_res.get("RMSE", None)
reported_mae = test_res.get("MAE",  None)
model_type   = meta.get("selected_model", "unknown")
split_method = "scaffold_split"  # evidenced by train/test_scaffolds keys

check_n    = reported_n    == 1240
check_r2   = reported_r2   is not None and abs(reported_r2   - 0.6156) < 1e-3
check_rmse = reported_rmse is not None and abs(reported_rmse - 0.7553) < 1e-3
check_mae  = reported_mae  is not None and abs(reported_mae  - 0.5879) < 1e-3

print(f"\nTest n=1240 confirmed:   {check_n}  (found: {reported_n})")
print(f"R²=0.6156 confirmed:     {check_r2}  (found: {reported_r2})")
print(f"RMSE=0.7553 confirmed:   {check_rmse} (found: {reported_rmse})")
print(f"MAE=0.5879 confirmed:    {check_mae}  (found: {reported_mae})")

# ── 3. Scaffold overlap check ─────────────────────────────────────────
train_scaffolds = meta.get("train_scaffolds", None)
test_scaffolds  = meta.get("test_scaffolds",  None)
val_scaffolds   = meta.get("validation_scaffolds", None)

# The metadata stores counts, not lists — check the training table instead
# The training table has a 'scaffold' column
print("\n" + "=" * 70)
print("SCAFFOLD OVERLAP CHECK")
print("=" * 70)

scaffold_overlap_found = None
duplicate_check_result = "no split labels in training table"

train_table = pd.read_csv(TRAIN_CSV)
print(f"Training table shape: {train_table.shape}")
print(f"Columns: {train_table.columns.tolist()}")
print(f"Sample pIC50 range: {train_table['pIC50'].min():.3f} – {train_table['pIC50'].max():.3f}")

# The metadata says train_scaffolds=2082, test_scaffolds=439; these are counts.
# Without a split label column in the training table we cannot directly
# check for scaffold overlap, but we can check scaffold counts.
if "scaffold" in train_table.columns:
    all_scaffolds  = set(train_table["scaffold"].dropna().tolist())
    print(f"Total unique scaffolds in table: {len(all_scaffolds)}")
    print(f"Metadata unique_scaffolds:       {meta.get('unique_scaffolds')}")

# Check for SMILES duplicates in the training table
dupes_smiles = train_table.duplicated(subset=["canonical_smiles"], keep=False)
n_dupes = dupes_smiles.sum()
print(f"\nDuplicate canonical SMILES in training table: {n_dupes}")

if n_dupes > 0:
    dup_rows = train_table[dupes_smiles][["canonical_smiles", "pIC50"]].head(10)
    print(dup_rows.to_string())
    duplicate_check_result = f"{n_dupes} duplicate canonical_smiles rows found"
else:
    duplicate_check_result = "no duplicate canonical_smiles in training table"

# Since the table has only canonical_smiles (no train/test split label),
# scaffold overlap cannot be directly confirmed from the table alone.
# We use metadata scaffold counts to reason about it.
n_train_scaf = meta.get("train_scaffolds", 0)
n_test_scaf  = meta.get("test_scaffolds",  0)
n_val_scaf   = meta.get("validation_scaffolds", 0)
n_total_scaf = meta.get("unique_scaffolds", 0)

# If sum of scaffold sets <= total unique, overlap could exist
sum_parts = n_train_scaf + n_test_scaf + n_val_scaf
print(f"\nScaffold count arithmetic:")
print(f"  train={n_train_scaf} + test={n_test_scaf} + val={n_val_scaf} = {sum_parts}")
print(f"  unique_scaffolds total = {n_total_scaf}")
print(f"  sum_parts - total = {sum_parts - n_total_scaf} (> 0 implies overlap)")

if sum_parts > n_total_scaf:
    scaffold_overlap_found = True
    overlap_count = sum_parts - n_total_scaf
    print(f"  -> Scaffold overlap DETECTED: ~{overlap_count} scaffolds shared across splits")
else:
    scaffold_overlap_found = False
    print("  -> No scaffold overlap detected from count arithmetic (counts are consistent with disjoint splits)")

# ── 4. Check for saved test predictions ─────────────────────────────────
print("\n" + "=" * 70)
print("TEST PREDICTION FILE SEARCH")
print("=" * 70)

test_pred_files = (
    list(MODEL_DIR.glob("*test_pred*")) +
    list(MODEL_DIR.glob("*predictions*")) +
    list(MODEL_DIR.glob("*test*pred*")) +
    list(MODEL_DIR.glob("*pred*test*"))
)
print(f"Files matching test_pred/predictions pattern: {test_pred_files}")

test_pearson_r   = None
test_spearman_rho = None

if test_pred_files:
    for f in test_pred_files:
        print(f"  Found: {f}")
        try:
            pred_df = pd.read_csv(f)
            print(f"  Columns: {pred_df.columns.tolist()}")
            # Try to compute metrics
            if "y_true" in pred_df.columns and "y_pred" in pred_df.columns:
                r, p   = stats.pearsonr(pred_df["y_true"], pred_df["y_pred"])
                rho, _ = stats.spearmanr(pred_df["y_true"], pred_df["y_pred"])
                test_pearson_r    = float(r)
                test_spearman_rho = float(rho)
                print(f"  Pearson r = {r:.4f}, Spearman rho = {rho:.4f}")
        except Exception as e:
            print(f"  Error reading {f}: {e}")
else:
    print("No saved test prediction files found in model directory.")
    print("Cannot compute Pearson r or Spearman rho from test predictions.")
    print("The training-set Pearson r (reported as ~0.901 in some audits)")
    print("MUST NOT be reported as external or test-set accuracy.")

# ── 5. pIC50 definition consistency ──────────────────────────────────────
print("\n" + "=" * 70)
print("pIC50 DEFINITION VERIFICATION")
print("=" * 70)

pic50_def = meta.get("pIC50_definition", "")
print(f"Metadata states: '{pic50_def}'")

# Check: if definition is '9 - log10(IC50[nM])', then pIC50=9 means IC50=1 nM
# pIC50=7 means IC50=100 nM (a reasonable BACE1 hit)
# pIC50=5 means IC50=10000 nM = 10 µM (weak binder)
# Training data range should be consistent with known BACE1 literature values.
pic50_min = train_table["pIC50"].min()
pic50_max = train_table["pIC50"].max()
pic50_mean = train_table["pIC50"].mean()

# Convert to IC50
ic50_min_nM  = 10**(9 - pic50_max)  # highest pIC50 = lowest IC50
ic50_max_nM  = 10**(9 - pic50_min)  # lowest pIC50 = highest IC50

print(f"Training pIC50 range: {pic50_min:.3f} - {pic50_max:.3f}")
print(f"  -> IC50 range: {ic50_min_nM:.2f} nM - {ic50_max_nM:.0f} nM")
print(f"  -> Mean pIC50: {pic50_mean:.3f}")

# BACE1 well-characterised range is ~0.1 nM to 100 uM, pIC50 ~4-10
# Some datasets include very weak or very potent outliers
definition_consistent = bool(
    "9 - log10" in pic50_def and
    pic50_min > 0.0 and    # pIC50 must be positive
    pic50_max < 13.0 and   # pIC50=12 means IC50=1 pM — unusual but not impossible
    ic50_min_nM > 0
)
print(f"pIC50 definition consistent with training data: {definition_consistent}")

# ── 6. Training-set Pearson r note ───────────────────────────────────────
training_pearson_note = (
    "The training-set Pearson r (~0.901) reflects in-sample fit only. "
    "It MUST NOT be reported as external or test-set accuracy. "
    "External (test-set) accuracy should be reported as R²=0.6156, "
    "RMSE=0.7553, MAE=0.5879 (n=1240 held-out test compounds)."
)

# ── 7. Assemble output JSON ───────────────────────────────────────────────
data_quality_notes = []
if n_dupes > 0:
    data_quality_notes.append(f"{n_dupes} duplicate canonical_smiles rows exist in training table")
if scaffold_overlap_found:
    data_quality_notes.append(
        f"Scaffold count arithmetic suggests ~{sum_parts - n_total_scaf} scaffolds "
        "shared across splits; genuine scaffold separation not fully confirmed"
    )
else:
    data_quality_notes.append(
        "Scaffold counts are consistent with disjoint scaffold splits "
        "(train+val+test scaffold counts sum to <= unique_scaffolds total)"
    )
if not test_pred_files:
    data_quality_notes.append(
        "No saved test prediction file found; Pearson r and Spearman rho on "
        "test set are NOT computable from available files"
    )

out = {
    "model_type":            model_type,
    "split_method":          split_method,
    "test_n":                int(reported_n) if reported_n is not None else None,
    "test_r2":               float(reported_r2) if reported_r2 is not None else None,
    "test_rmse":             float(reported_rmse) if reported_rmse is not None else None,
    "test_mae":              float(reported_mae) if reported_mae is not None else None,
    "test_pearson_r":        float(test_pearson_r) if test_pearson_r is not None else None,
    "test_spearman_rho":     float(test_spearman_rho) if test_spearman_rho is not None else None,
    "training_pearson_note": training_pearson_note,
    "scaffold_overlap_found": bool(scaffold_overlap_found) if scaffold_overlap_found is not None else None,
    "pic50_definition":      pic50_def,
    "pic50_definition_consistent": bool(definition_consistent),
    "pic50_training_range":  [float(pic50_min), float(pic50_max)],
    "ic50_equivalent_range_nM": [float(ic50_min_nM), float(ic50_max_nM)],
    "duplicate_check_result": duplicate_check_result,
    "metadata_checks": {
        "test_n_confirmed":    bool(check_n),
        "r2_confirmed":        bool(check_r2),
        "rmse_confirmed":      bool(check_rmse),
        "mae_confirmed":       bool(check_mae),
    },
    "data_quality_notes":    data_quality_notes,
}

OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\nWritten: {OUT_JSON}")
print(json.dumps(out, indent=2))
