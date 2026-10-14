"""
Phase 2 — Authoritative Master Candidate Table
Builds results/bace1_master_candidate_table.csv covering all 11,376 oracle-evaluated molecules.
Also saves results/bace1_candidate_exclusions.csv from oracle_exclusion_reconciliation.csv.
"""

import os
import sys
import warnings
import logging

# Suppress all Python warnings
warnings.filterwarnings("ignore")
logging.getLogger("rdkit").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs

# Silence RDKit C++ deprecation warnings to stderr
RDLogger.DisableLog("rdApp.*")

from pathlib import Path

# Route all output to a log file so nothing is lost
_SCRIPTS_DIR = Path(__file__).resolve().parent
_LOG_PATH = _SCRIPTS_DIR / "phase2_master_out.txt"
_log = open(_LOG_PATH, "w", encoding="utf-8", buffering=1)
sys.stdout = _log
# Keep stderr going to real stderr but also log it
import io
class _TeeStderr(io.TextIOBase):
    def __init__(self, real, log):
        self._real = real
        self._log = log
    def write(self, s):
        # Only suppress RDKit deprecation noise
        if "DEPRECATION WARNING" not in s and "rdApp" not in s:
            self._real.write(s)
            self._log.write(s)
        return len(s)
    def flush(self):
        self._real.flush()
        self._log.flush()
sys.stderr = _TeeStderr(sys.__stderr__, _log)

# Use new MorganGenerator API to avoid DEPRECATION WARNING spam
try:
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    _morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    def morgan_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return _morgan_gen.GetFingerprint(mol)
except ImportError:
    from rdkit.Chem import AllChem
    def morgan_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

BASE = str(Path(__file__).resolve().parent.parent)


def path(rel):
    return os.path.join(BASE, rel)


def canonicalize(smiles):
    """Return canonical SMILES or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# morgan_fp is defined above using new API or fallback


# ---------------------------------------------------------------------------
# 1. Load base table
# ---------------------------------------------------------------------------
print("Loading oracle_pool_16384_pred.csv …")
base_df = pd.read_csv(path("results/oracle_pool_16384_pred.csv"))
print(f"  Loaded {len(base_df):,} rows, {len(base_df.columns)} columns")

# Canonicalize base SMILES (column already exists as 'canonical_smiles')
# but re-canonicalize to ensure RDKit consistency for merging
print("  Re-canonicalizing SMILES with RDKit …")
base_df["canon_key"] = base_df["canonical_smiles"].apply(canonicalize)
n_failed = base_df["canon_key"].isna().sum()
print(f"  Failed canonicalization: {n_failed} rows")

assert len(base_df) == 11376, f"Expected 11,376 rows, got {len(base_df)}"

# Rename columns for clarity in the master table
base_df = base_df.rename(columns={
    "predicted_pIC50": "oracle_predicted_pIC50",
    "predicted_logBB": "oracle_predicted_logBB",
})

# ---------------------------------------------------------------------------
# 2. Left-join top200 scaffold/QED/Ro5/LogP/MW/TPSA/HBD
# ---------------------------------------------------------------------------
print("\nLoading top200_scaffold_analysis.csv …")
scaffold_df = pd.read_csv(path("results/top200_scaffold_analysis.csv"))
scaffold_df["canon_key"] = scaffold_df["canonical_smiles"].apply(canonicalize)
# Note: QED is not in this file; it's only available in bace1_validated_candidates.csv
# We bring in all available columns from top200_scaffold_analysis
scaffold_cols = ["canon_key", "MW", "LogP", "HBD", "HBA", "TPSA", "RotB",
                 "Rings", "ArRings", "Ro5", "Veber", "scaffold"]
scaffold_df = scaffold_df[scaffold_cols].copy()
scaffold_df = scaffold_df.rename(columns={c: f"t200_{c}" for c in scaffold_cols if c != "canon_key"})

# deduplicate on key (should be 200 unique)
scaffold_df = scaffold_df.drop_duplicates(subset=["canon_key"])
print(f"  {len(scaffold_df):,} unique canonical keys in top200 scaffold table")

base_df = base_df.merge(scaffold_df, on="canon_key", how="left")
n_top200_joined = base_df["t200_scaffold"].notna().sum()
print(f"  Joined top200 scaffold data for {n_top200_joined} rows (expected 200)")

# ---------------------------------------------------------------------------
# 3. Left-join docking data from docking_reconciliation.csv
# ---------------------------------------------------------------------------
print("\nLoading docking_reconciliation.csv …")
dock_df = pd.read_csv(path("reports/audit_corrections/docking_reconciliation.csv"))
dock_df["canon_key"] = dock_df["canonical_smiles"].apply(canonicalize)

# Use parsed_vina_score as the authoritative score
dock_cols = ["canon_key", "parsed_vina_score", "pose_file", "rank"]
dock_df_sub = dock_df[dock_cols].copy()
dock_df_sub = dock_df_sub.rename(columns={
    "parsed_vina_score": "vina_score",
    "rank": "docking_rank",
})
dock_df_sub["docking_status"] = "docked"
dock_df_sub = dock_df_sub.drop_duplicates(subset=["canon_key"])
print(f"  {len(dock_df_sub):,} docked molecules (expected 50)")

base_df = base_df.merge(dock_df_sub, on="canon_key", how="left")
n_docked = base_df["docking_status"].notna().sum()
print(f"  Joined docking data for {n_docked} rows (expected 50)")

# ---------------------------------------------------------------------------
# 4. Left-join PAINS/Brenk/Reactive/verdict from bace1_validated_candidates.csv
# ---------------------------------------------------------------------------
print("\nLoading bace1_validated_candidates.csv …")
validated_df = pd.read_csv(path("results/docking/bace1_validated_candidates.csv"))
validated_df["canon_key"] = validated_df["canonical_smiles"].apply(canonicalize)

validated_cols = ["canon_key", "PAINS", "Brenk", "Reactive", "n_alerts",
                  "max_sim_known_BACE1", "verdict", "QED"]
validated_df_sub = validated_df[validated_cols].copy()
validated_df_sub = validated_df_sub.rename(columns={c: f"chem_{c}" for c in validated_cols if c != "canon_key"})
validated_df_sub = validated_df_sub.drop_duplicates(subset=["canon_key"])
print(f"  {len(validated_df_sub):,} validated candidates (expected 11)")

base_df = base_df.merge(validated_df_sub, on="canon_key", how="left")
n_validated = base_df["chem_verdict"].notna().sum()
print(f"  Joined chemistry filter data for {n_validated} rows (expected 11)")

# ---------------------------------------------------------------------------
# 5. Compute derived Boolean columns
# ---------------------------------------------------------------------------
print("\nComputing derived Boolean columns …")

base_df["passes_pIC50_7_6576"] = base_df["oracle_predicted_pIC50"] >= 7.6576
base_df["passes_logBB_0"] = base_df["oracle_predicted_logBB"] >= 0.0
base_df["passes_original_joint_threshold"] = (
    base_df["passes_pIC50_7_6576"] & base_df["passes_logBB_0"]
)

# passes_BBB_and_docking: only defined where docking data exists
base_df["passes_BBB_and_docking"] = np.where(
    base_df["docking_status"].notna(),
    (base_df["oracle_predicted_logBB"] >= 0.0) & (base_df["vina_score"] <= -7.5),
    np.nan,
)
# Cast to object so we can have bool/NaN
base_df["passes_BBB_and_docking"] = base_df["passes_BBB_and_docking"].astype(object)
mask_docked = base_df["docking_status"].notna()
base_df.loc[mask_docked, "passes_BBB_and_docking"] = (
    (base_df.loc[mask_docked, "oracle_predicted_logBB"] >= 0.0) &
    (base_df.loc[mask_docked, "vina_score"] <= -7.5)
)

# passes_all_chemistry_filters: only for the 11 validated rows
# True if verdict in ['CLEAN', 'CLEAN (low QED)'], False if other verdict, NA otherwise
clean_verdicts = {"CLEAN", "CLEAN (low QED)"}
base_df["passes_all_chemistry_filters"] = np.nan
base_df["passes_all_chemistry_filters"] = base_df["passes_all_chemistry_filters"].astype(object)
mask_validated = base_df["chem_verdict"].notna()
base_df.loc[mask_validated, "passes_all_chemistry_filters"] = (
    base_df.loc[mask_validated, "chem_verdict"].isin(clean_verdicts)
)

# ---------------------------------------------------------------------------
# 6. Morgan ECFP4 Tanimoto similarity to training set (top 200 only)
# ---------------------------------------------------------------------------
print("\nComputing Tanimoto similarity for top-200 candidates …")
training_df = pd.read_csv(path("results/bace1_activity_model/bace1_activity_training_table.csv"))
print(f"  Training set size: {len(training_df):,}")

# Pre-compute training fingerprints
print("  Computing training fingerprints …")
train_fps = []
train_smiles_list = []
for smi in training_df["canonical_smiles"]:
    fp = morgan_fp(smi)
    if fp is not None:
        train_fps.append(fp)
        train_smiles_list.append(smi)
print(f"  Valid training fingerprints: {len(train_fps):,}")

# Compute for top-200 rows
# Initialize as object dtype to allow mixed float NaN and string values
base_df["max_training_Tc"] = pd.Series(np.nan, index=base_df.index, dtype=object)
base_df["nearest_training_smiles"] = pd.Series(np.nan, index=base_df.index, dtype=object)
base_df["exact_training_duplicate"] = pd.Series(np.nan, index=base_df.index, dtype=object)

top200_mask = base_df["t200_scaffold"].notna()
top200_idx = base_df.index[top200_mask]
print(f"  Processing {len(top200_idx)} top-200 molecules …")

for i, idx in enumerate(top200_idx):
    if (i + 1) % 50 == 0:
        print(f"    {i + 1}/{len(top200_idx)} done …")
    smi = base_df.loc[idx, "canon_key"]
    fp = morgan_fp(smi)
    if fp is None:
        continue
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    max_sim = float(np.max(sims))
    best_idx = int(np.argmax(sims))
    base_df.at[idx, "max_training_Tc"] = max_sim
    base_df.at[idx, "nearest_training_smiles"] = train_smiles_list[best_idx]
    base_df.at[idx, "exact_training_duplicate"] = max_sim >= 0.999

print("  Tanimoto computation complete.")

# ---------------------------------------------------------------------------
# 7. Verify two key candidates
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("KEY CANDIDATE VERIFICATION")
print("=" * 60)

cand1_smi = canonicalize("CC(C)(C)COc1cnc2c(c1)C1(COC(N)=N1)c1cc(-c3cncnc3)ccc1O2")
cand2_smi = canonicalize("CC1CCC(COc2ccc(F)cn2)CN1C(O)c1cc(F)ccc1-c1ncccn1")

for label, query_canon, exp_pic50, exp_logBB in [
    ("Candidate 1 (expected pIC50≈7.9037, logBB≈-0.020)", cand1_smi, 7.9037, -0.020),
    ("Candidate 2 (expected pIC50≈6.169, logBB≈+0.023)", cand2_smi, 6.169, 0.023),
]:
    row = base_df[base_df["canon_key"] == query_canon]
    print(f"\n{label}")
    if len(row) == 0:
        print("  NOT FOUND in master table")
    else:
        r = row.iloc[0]
        print(f"  Found: canon_key = {r['canon_key']}")
        print(f"  oracle_predicted_pIC50 = {r['oracle_predicted_pIC50']:.6f}  (expected ≈{exp_pic50})")
        print(f"  oracle_predicted_logBB = {r['oracle_predicted_logBB']:.6f}  (expected ≈{exp_logBB})")
        if pd.notna(r.get("vina_score")):
            print(f"  vina_score = {r['vina_score']:.3f}")
        if pd.notna(r.get("chem_verdict")):
            print(f"  chem_verdict = {r['chem_verdict']}")
        if pd.notna(r.get("passes_BBB_and_docking")):
            print(f"  passes_BBB_and_docking = {r['passes_BBB_and_docking']}")
        if pd.notna(r.get("max_training_Tc")):
            print(f"  max_training_Tc = {r['max_training_Tc']:.4f}")
            print(f"  exact_training_duplicate = {r['exact_training_duplicate']}")

# ---------------------------------------------------------------------------
# 8. Summary counts
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY COUNTS")
print("=" * 60)

total = len(base_df)
print(f"\nTotal rows: {total:,}  (expected 11,376)")

n_pic50 = base_df["passes_pIC50_7_6576"].sum()
print(f"passes_pIC50_7_6576: {n_pic50:,} / {total:,}  ({n_pic50/total:.1%})")

n_logbb = base_df["passes_logBB_0"].sum()
print(f"passes_logBB_0: {n_logbb:,} / {total:,}  ({n_logbb/total:.1%})")

n_joint = base_df["passes_original_joint_threshold"].sum()
print(f"passes_original_joint_threshold (BOTH): {n_joint:,}  (expected ZERO)")

# passes_BBB_and_docking: count True values
n_bbb_dock = (base_df["passes_BBB_and_docking"] == True).sum()
print(f"passes_BBB_and_docking = True: {n_bbb_dock:,}  (expected 11)")

# passes_all_chemistry_filters = True
n_clean = (base_df["passes_all_chemistry_filters"] == True).sum()
print(f"passes_all_chemistry_filters = True: {n_clean:,}  (expected 2)")

# High oracle pIC50 (>=7.0) failing BBB (logBB<0)
n_high_pic50_fail_bbb = (
    (base_df["oracle_predicted_pIC50"] >= 7.0) &
    (base_df["oracle_predicted_logBB"] < 0.0)
).sum()
print(f"High pIC50 (>=7.0) failing BBB (logBB<0): {n_high_pic50_fail_bbb:,}  (expected ~5)")

# ---------------------------------------------------------------------------
# 9. Save master table
# ---------------------------------------------------------------------------
print("\nSaving master table …")
out_master = path("results/bace1_master_candidate_table.csv")
base_df = base_df.drop(columns=["canon_key"])  # internal merge key, not needed in output
base_df.to_csv(out_master, index=False)
print(f"  Saved: {out_master}")
print(f"  Shape: {base_df.shape}")

# ---------------------------------------------------------------------------
# 10. Save exclusions table
# ---------------------------------------------------------------------------
print("\nSaving exclusions table …")
excl_df = pd.read_csv(path("reports/audit_corrections/oracle_exclusion_reconciliation.csv"))
# Reformat: rename columns to be more descriptive, add exclusion_category
excl_out = excl_df.rename(columns={
    "source_record_id": "source_sample_id",
    "original_smiles": "original_smiles",
    "canonical_smiles_if_available": "canonical_smiles",
    "valid_flag": "rdkit_valid",
    "exclusion_reason": "exclusion_reason",
})
# Parse duplicate pattern to extract first-seen record
import re
excl_out["exclusion_category"] = excl_out["exclusion_reason"].apply(
    lambda r: "duplicate_canonical_smiles" if "duplicate" in str(r).lower() else str(r)
)
excl_out["first_seen_record_id"] = excl_out["exclusion_reason"].apply(
    lambda r: int(m.group(1)) if (m := re.search(r"first seen at record (\d+)", str(r))) else None
)

out_excl = path("results/bace1_candidate_exclusions.csv")
excl_out.to_csv(out_excl, index=False)
print(f"  Saved: {out_excl}")
print(f"  Shape: {excl_out.shape}")

# ---------------------------------------------------------------------------
# Final confirmation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("=== PHASE 2 COMPLETE ===")
print("=" * 60)
print(f"\nMaster candidate table:")
print(f"  {out_master}")
print(f"  {len(base_df):,} rows × {len(base_df.columns)} columns")
print(f"\nCandidate exclusions table:")
print(f"  {out_excl}")
print(f"  {len(excl_out):,} rows × {len(excl_out.columns)} columns")
print(f"\nVerified counts:")
print(f"  Total rows:                     {total:,}")
print(f"  passes_pIC50_7_6576:            {n_pic50:,}  ({n_pic50/total:.1%})")
print(f"  passes_logBB_0:                 {n_logbb:,}  ({n_logbb/total:.1%})")
print(f"  passes_original_joint:          {n_joint:,}  (expected 0)")
print(f"  passes_BBB_and_docking=True:    {n_bbb_dock:,}  (expected 11)")
print(f"  passes_all_chem_filters=True:   {n_clean:,}  (expected 2)")
print(f"  High pIC50 failing BBB:         {n_high_pic50_fail_bbb:,}  (expected ~5)")

# Restore stdout/stderr and close log file
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
_log.close()
print("Phase 2 script complete. Output written to scripts/phase2_master_out.txt")
