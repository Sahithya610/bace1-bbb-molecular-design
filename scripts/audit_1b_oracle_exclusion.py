"""
Audit 1B: Oracle Exclusion Reconciliation (292 missing records)
Determines why 11,668 valid=True rows in oracle_pool_16384.csv
produced only 11,376 rows in oracle_pool_16384_pred.csv.
"""

import sys
import warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# ── Paths ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
POOL_CSV   = ROOT / "results" / "oracle_pool_16384.csv"
PRED_CSV   = ROOT / "results" / "oracle_pool_16384_pred.csv"
OUT_DIR    = ROOT / "reports" / "audit_corrections"
OUT_CSV    = OUT_DIR / "oracle_exclusion_reconciliation.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# predict_joint.py constants (from reading the actual script)
RADIUS = 2
N_BITS = 2048

# ── Load data ───────────────────────────────────────────────────────────
print("Loading pool CSV …")
pool_df = pd.read_csv(POOL_CSV)
print(f"  Total pool rows:  {len(pool_df)}")

valid_df = pool_df[pool_df["valid"] == True].copy().reset_index(drop=True)
print(f"  valid=True rows:  {len(valid_df)}")

print("Loading pred CSV …")
pred_df = pd.read_csv(PRED_CSV)
print(f"  Pred rows:        {len(pred_df)}")

# Collect the set of canonical SMILES present in the predictions
pred_canon = set(pred_df["canonical_smiles"].dropna().tolist())
print(f"  Unique pred canonical_smiles: {len(pred_canon)}")

# ── Helper: canonicalize ─────────────────────────────────────────────────
def safe_canonicalize(smi):
    """Returns canonical SMILES string or None on failure, plus error tag."""
    if not isinstance(smi, str) or not smi.strip():
        return None, "empty_or_non_string"
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "rdkit_parse_failure"
    try:
        canon = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None, "canonical_smiles_failure"
    return canon, None

def safe_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "rdkit_parse_failure_fp"
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=RADIUS, nBits=N_BITS)
        return fp, None
    except Exception as e:
        return None, f"fingerprint_failure: {e}"

# ── Canonicalize all valid rows ────────────────────────────────────────────
print("Canonicalising valid rows …")
canon_list   = []
canon_errors = []
for smi in valid_df["smiles"]:
    c, err = safe_canonicalize(smi)
    canon_list.append(c)
    canon_errors.append(err)

valid_df["_canon"]       = canon_list
valid_df["_canon_error"] = canon_errors

# ── Identify which valid rows ARE in pred (by canonical SMILES) ──────────
valid_df["_in_pred"] = valid_df["_canon"].apply(
    lambda c: (c is not None) and (c in pred_canon)
)
print(f"  Valid rows found in pred by canonical SMILES: {valid_df['_in_pred'].sum()}")

# ── For rows NOT in pred, determine the exclusion reason ─────────────────
# Also track first-seen canonical SMILES to detect duplicates
seen_canon = {}  # canon_smiles -> first source_record_id

exclusion_records = []
reasons_count = {}

for idx, row in valid_df.iterrows():
    original_smi = row["smiles"]
    canon        = row["_canon"]
    canon_err    = row["_canon_error"]
    source_id    = row.get("sample_id", idx)
    in_pred      = row["_in_pred"]

    if in_pred:
        # Not excluded — skip
        continue

    # Determine exclusion reason
    reason = None

    # a. RDKit cannot parse at all
    if canon is None and canon_err in ("rdkit_parse_failure", "empty_or_non_string"):
        reason = f"rdkit_parse_failure: {canon_err}"

    # b. RDKit parses but canonical SMILES fails
    elif canon is None and canon_err == "canonical_smiles_failure":
        reason = "canonical_smiles_computation_failure"

    # c. Canonical SMILES computable — check fingerprint
    elif canon is not None:
        fp, fp_err = safe_fingerprint(canon)
        if fp is None:
            reason = f"fingerprint_failure: {fp_err}"
        else:
            # d. Check for duplicate canonical SMILES
            if canon in seen_canon:
                reason = f"duplicate_canonical_smiles (earlier record: {seen_canon[canon]})"
            else:
                # e. Must have been filtered by predict_joint.py's drop_duplicates
                #    (predict_joint.py canonicalises then drops duplicates)
                #    If we reach here, the molecule is chemically valid but
                #    its canonical form was seen earlier in the iteration order.
                #    Mark as duplicate-in-pipeline.
                reason = "duplicate_removed_by_drop_duplicates_in_predict_joint"

    else:
        reason = f"unknown: {canon_err}"

    # Record first-seen for dup tracking (even for excluded rows)
    if canon is not None and canon not in seen_canon:
        seen_canon[canon] = source_id

    reasons_count[reason] = reasons_count.get(reason, 0) + 1

    exclusion_records.append({
        "source_record_id":            source_id,
        "original_smiles":             original_smi,
        "canonical_smiles_if_available": canon,
        "valid_flag":                  row["valid"],
        "exclusion_reason":            reason,
    })

# ── But we also need to track seen_canon for non-excluded valid rows
#    to correctly attribute duplicate exclusions.
# Re-run in order, tracking all valid rows

print("\nRe-running in order to correctly detect duplicates …")
seen_canon2     = {}
exclusion_rows2 = []
reasons_count2  = {}
excluded_count  = 0
included_count  = 0

for idx, row in valid_df.iterrows():
    original_smi = row["smiles"]
    source_id    = row.get("sample_id", idx)

    # Step 1: canonicalize
    canon, canon_err = safe_canonicalize(original_smi)

    if canon is None:
        reason = f"rdkit_failure: {canon_err}"
        excluded_count += 1
    else:
        # Step 2: fingerprint
        fp, fp_err = safe_fingerprint(canon)
        if fp is None:
            reason = f"fingerprint_failure: {fp_err}"
            excluded_count += 1
        else:
            # Step 3: duplicate check (predict_joint.py drops on canonical_smiles)
            if canon in seen_canon2:
                reason = f"duplicate_canonical_smiles (first seen at record {seen_canon2[canon]})"
                excluded_count += 1
            else:
                # Not excluded by pipeline — mark as included
                seen_canon2[canon] = source_id
                included_count += 1
                continue  # Not an exclusion, skip

    # Track first seen even for failed (for dup detection)
    if canon is not None and canon not in seen_canon2:
        seen_canon2[canon] = source_id

    reasons_count2[reason] = reasons_count2.get(reason, 0) + 1
    exclusion_rows2.append({
        "source_record_id":              source_id,
        "original_smiles":               original_smi,
        "canonical_smiles_if_available": canon,
        "valid_flag":                    row["valid"],
        "exclusion_reason":              reason,
    })

print(f"\n  Included (pipeline would keep): {included_count}")
print(f"  Excluded (pipeline would drop): {excluded_count}")
print(f"\n  Exclusion reason breakdown:")
for reason, cnt in sorted(reasons_count2.items(), key=lambda x: -x[1]):
    print(f"    {cnt:5d}  {reason}")

# ── Cross-check: pipeline-included count vs pred CSV length ──────────────
print(f"\n  Pred CSV rows (actual):        {len(pred_df)}")
print(f"  Pipeline included (computed):  {included_count}")
delta = included_count - len(pred_df)
print(f"  Delta:                         {delta}")
if abs(delta) > 5:
    print("  WARNING: computed inclusion count differs from pred CSV rows by > 5.")
    print("  There may be additional filters in predict_joint.py not captured here.")

# ── Save output CSV ───────────────────────────────────────────────────────
df_excl = pd.DataFrame(exclusion_rows2)
df_excl.to_csv(OUT_CSV, index=False)
print(f"\nWritten: {OUT_CSV}  ({len(df_excl)} rows)")
