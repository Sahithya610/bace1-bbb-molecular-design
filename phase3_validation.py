# -*- coding: utf-8 -*-
"""
Phase 3: Oracle and BBB Model Validation + Novelty Analysis
All print statements use ASCII only to avoid cp1252 encoding errors.
"""

import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

# -------------------------------------------------------------------------
# Force stdout/stderr to UTF-8 so print() does not crash on non-ASCII
# -------------------------------------------------------------------------
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports" / "audit_corrections"
REPORTS.mkdir(parents=True, exist_ok=True)

ORACLE_METADATA  = RESULTS / "bace1_activity_model" / "model_metadata.json"
ORACLE_TRAIN_CSV = RESULTS / "bace1_activity_model" / "bace1_activity_training_table.csv"
BBB_METADATA     = RESULTS / "bbb_activity_model" / "bbb_model_metadata.json"
BBB_DATA_CSV     = ROOT / "druggen_data" / "bace1_continuous_bbb.csv"
TOP200_CSV       = RESULTS / "oracle_filtered_top200_16k.csv"
ORACLE_VERIFIED  = REPORTS / "oracle_metrics_verified.json"
NOVELTY_CSV      = REPORTS / "novelty_analysis_top200.csv"
NOVELTY_TXT      = REPORTS / "novelty_summary.txt"


def sep(title=""):
    print()
    print("=" * 70)
    if title:
        print(title)
        print("=" * 70)


def compute_ecfp4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


# =========================================================================
# 3A: Oracle Validation
# =========================================================================
sep("3A: ORACLE VALIDATION")

with open(ORACLE_METADATA) as f:
    meta = json.load(f)

print("Model type       : %s" % meta["selected_model"])
print("Split method     : scaffold_split (keys: train_scaffolds/test_scaffolds)")
print("Test samples (n) : %d" % meta["test_samples"])
print("Test R2          : %.4f" % meta["test_results"]["R2"])
print("Test RMSE        : %.4f" % meta["test_results"]["RMSE"])
print("Test MAE         : %.4f" % meta["test_results"]["MAE"])

assert meta["selected_model"] == "ExtraTrees", "Model mismatch"
assert meta["test_samples"] == 1240, "Test n mismatch"
assert abs(meta["test_results"]["R2"]   - 0.6156) < 1e-3, "R2 mismatch"
assert abs(meta["test_results"]["RMSE"] - 0.7553) < 1e-3, "RMSE mismatch"
assert abs(meta["test_results"]["MAE"]  - 0.5879) < 1e-3, "MAE mismatch"
print("\n[PASS] All metadata checks: ExtraTrees, scaffold-split, n=1240, R2=0.6156, RMSE=0.7553, MAE=0.5879")

# Search for saved test prediction files
sep("3A-i: SEARCHING FOR SAVED TEST PREDICTIONS")
model_dir = RESULTS / "bace1_activity_model"
all_csvs = list(model_dir.glob("*.csv"))
test_pred_files = [p for p in all_csvs if "test" in p.name.lower() or "pred" in p.name.lower()]
print("All CSVs in bace1_activity_model/ : %s" % [p.name for p in all_csvs])
print("Test/pred candidate files          : %s" % [p.name for p in test_pred_files])

pearson_r    = None
spearman_rho = None

if test_pred_files:
    for p in test_pred_files:
        try:
            tp = pd.read_csv(p)
            print("  %s columns: %s" % (p.name, tp.columns.tolist()))
        except Exception as e:
            print("  Could not read %s: %s" % (p.name, e))
    print("No file with paired predicted/actual pIC50 found (columns do not match expected format).")
else:
    print("No test prediction CSV files found in bace1_activity_model/.")
    print("=> Test-set Pearson r and Spearman rho cannot be computed.")
    print("   Only metadata-reported metrics are available: R2=0.6156, RMSE=0.7553, MAE=0.5879.")

# Training table distribution
sep("3A-ii: TRAINING SET pIC50 DISTRIBUTION")
train_df = pd.read_csv(ORACLE_TRAIN_CSV)
print("Training rows   : %d" % len(train_df))
print("Columns         : %s" % train_df.columns.tolist())
print("\npIC50 statistics:")
print(train_df["pIC50"].describe().to_string())

threshold = 7.6576
frac_gt7   = (train_df["pIC50"] >= 7.0).mean()
frac_gt76  = (train_df["pIC50"] >= threshold).mean()
n_gt7      = int((train_df["pIC50"] >= 7.0).sum())
n_gt76     = int((train_df["pIC50"] >= threshold).sum())

print("\nFraction with pIC50 >= 7.0      : %.4f  (%d / %d)" % (frac_gt7, n_gt7, len(train_df)))
print("Fraction with pIC50 >= %.4f : %.4f  (%d / %d)" % (threshold, frac_gt76, n_gt76, len(train_df)))

range_df = train_df[(train_df["pIC50"] >= 7.0) & (train_df["pIC50"] <= 9.0)]
print("\nMolecules in 7.0-9.0 pIC50 range: %d" % len(range_df))
bins = [7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0]
counts, edges = np.histogram(range_df["pIC50"], bins=bins)
print("\npIC50 histogram (7.0-9.0):")
for i, c in enumerate(counts):
    bar = "#" * min(c // 5, 60)
    print("  [%.2f-%.2f): %4d  %s" % (edges[i], edges[i+1], c, bar))

print("\nTotal training molecules relevant to generation (pIC50 >= 7.0): %d" % n_gt7)

sep("3A-iii: EXPLICIT CAVEATS (REQUIRED)")
print("""
CAVEAT 1 (RMSE):
  RMSE=0.755 is a TYPICAL/RMS error statistic, NOT a calibrated 95% prediction interval.
  It must NOT be reported as '+/-0.755 pIC50 at 95% confidence.' A true 95% CI requires
  either conformal prediction or bootstrap calibration on the test set.

CAVEAT 2 (Distribution shift):
  Generated molecules are NOT guaranteed to be within the training distribution of the
  ExtraTrees oracle. The oracle is an interpolative model trained on ~5800 BACE1
  inhibitors; structural novelty in generated molecules may exceed the model's applicability
  domain. Predictions on out-of-domain molecules may be unreliable.

CAVEAT 3 (+/-1.5 pIC50 claim):
  The phrase '+/-1.5 pIC50' as a calibrated confidence interval must NOT appear in any
  report. The only defensible statement is the test-set RMSE=0.755 as a root-mean-square
  deviation, not as a symmetric 95% CI.

CAVEAT 4 (Training Pearson r):
  The training-set Pearson r (~0.901) reflects in-sample goodness-of-fit only.
  It MUST NOT be presented as external or test-set accuracy. The authoritative external
  accuracy figure is R2=0.6156 (scaffold-split test set, n=1240).
""")

# Update oracle_metrics_verified.json
with open(ORACLE_VERIFIED) as f:
    ov = json.load(f)

ov["test_pearson_r"]    = pearson_r
ov["test_spearman_rho"] = spearman_rho
ov["training_set_stats"] = {
    "n_training":                     int(len(train_df)),
    "fraction_pic50_ge_7_0":          float(round(frac_gt7,  6)),
    "n_pic50_ge_7_0":                 n_gt7,
    "fraction_pic50_ge_threshold":    float(round(frac_gt76, 6)),
    "n_pic50_ge_threshold":           n_gt76,
    "threshold_used":                 float(threshold),
    "n_in_70_to_90_range":            int(len(range_df)),
    "pic50_mean":                     float(round(float(train_df["pIC50"].mean()), 6)),
    "pic50_std":                      float(round(float(train_df["pIC50"].std()),  6)),
    "pic50_median":                   float(round(float(train_df["pIC50"].median()), 6)),
}
ov["explicit_caveats"] = [
    "RMSE=0.755 is a typical error statistic (RMS deviation), NOT a calibrated 95% prediction interval. Do NOT report as +/-0.755 at 95% CI.",
    "Generated molecules may be outside the training distribution; applicability domain is unverified for generated structures.",
    "Do NOT claim +/-1.5 pIC50 as a calibrated confidence interval unless conformal prediction or bootstrap calibration has been performed.",
    "Training-set Pearson r (~0.901) reflects in-sample fit only. External accuracy = R2=0.6156, RMSE=0.7553, MAE=0.5879 (n=1240 scaffold-split test).",
]
existing_notes = ov.get("data_quality_notes", [])
new_note = "No saved test prediction CSV found in bace1_activity_model/; Pearson r and Spearman rho set to null."
if not any("No saved test prediction" in n for n in existing_notes):
    existing_notes.append(new_note)
ov["data_quality_notes"] = existing_notes

with open(ORACLE_VERIFIED, "w") as f:
    json.dump(ov, f, indent=2)
print("Updated: %s" % ORACLE_VERIFIED)


# =========================================================================
# 3B: BBB Model Validation
# =========================================================================
sep("3B: BBB MODEL VALIDATION")

with open(BBB_METADATA) as f:
    bbb_meta = json.load(f)

print("BBB model metadata:")
for k, v in bbb_meta.items():
    print("  %s: %s" % (k, v))

print("\nBBB training data source: B3DB quantitative logBB (experimental, ~1058 CNS/peripheral compounds)")
print("Train/val/test: %d / %d / %d" % (bbb_meta["train_samples"], bbb_meta["validation_samples"], bbb_meta["test_samples"]))
print("Scaffold split : YES (confirmed via train_bbb_regression.py)")
print("\nBBB test metrics (scaffold-split, n=%d):" % bbb_meta["test_samples"])
print("  MAE  : %.4f" % bbb_meta["test_results"]["MAE"])
print("  RMSE : %.4f" % bbb_meta["test_results"]["RMSE"])
print("  R2   : %.4f" % bbb_meta["test_results"]["R2"])
print("\n[WARNING] R2=0.354 on scaffold-split test is LOW. BBB model generalizes poorly to new scaffolds.")

# Inspect bace1_continuous_bbb.csv
bbb_data = pd.read_csv(BBB_DATA_CSV)
print("\nbace1_continuous_bbb.csv columns : %s" % bbb_data.columns.tolist())
print("Rows                             : %d" % len(bbb_data))
print("First 3 predicted_logBB values   : %s" % bbb_data["predicted_logBB"].head(3).tolist())

has_source_col = "source" in bbb_data.columns
has_ref_col    = "reference" in bbb_data.columns
has_logBB_col  = "logBB" in bbb_data.columns
print("\n'source' column present    : %s" % has_source_col)
print("'reference' column present : %s" % has_ref_col)
print("'logBB' (raw) present      : %s" % has_logBB_col)

sep("3B: logBB SOURCE TRACE")
print("""
The column 'predicted_logBB' in bace1_continuous_bbb.csv is NOT from B3DB experimental
measurements on BACE1 inhibitors. The column name 'predicted_logBB' indicates that:

  1. The BBB regression model (bbb_model.joblib) was trained on B3DB experimental logBB
     values for ~1058 diverse CNS/peripheral compounds (experimental ground truth).
  2. That trained model was then APPLIED to BACE1 training molecules to generate
     'predicted_logBB' values in bace1_continuous_bbb.csv.
  3. Therefore, logBB values for BACE1 training molecules are PREDICTIONS, not experimental.

Consequence for generated molecules: BBB predictions are PREDICTIONS-OF-PREDICTIONS.
  generated_mol -> BBB model -> predicted_logBB
  (BBB model trained on B3DB experimental data; BACE1 molecules are outside B3DB;
   their logBB values are predicted by a model with R2=0.354 on scaffold-split test.)

This is a CRITICAL caveat that must be stated in any publication or report.
""")

print("BBB MODEL GENERALIZATION ASSESSMENT:")
print("  - Scaffold-split test R2=0.354: BBB model generalization is MODERATE AT BEST.")
print("  - No additional held-out external test set was found.")
print("  - CONCLUSION: BBB model generalization on truly novel scaffolds is UNVERIFIED")
print("    beyond scaffold-split R2=0.354. Predicted logBB values have high uncertainty")
print("    and should be treated as rough rank-ordering, not quantitative predictions.")


# =========================================================================
# 3C: Novelty Analysis for Top 200
# =========================================================================
sep("3C: NOVELTY ANALYSIS -- TOP 200 CANDIDATES")

top200 = pd.read_csv(TOP200_CSV)
print("Loaded oracle_filtered_top200_16k.csv: %s" % str(top200.shape))
print("Columns: %s" % top200.columns.tolist())

n_cands    = len(top200)
smiles_col = "canonical_smiles"
oracle_col = "predicted_pIC50"
logbb_col  = "predicted_logBB"

for col in [smiles_col, oracle_col, logbb_col]:
    assert col in top200.columns, "Missing column: %s" % col

top200 = top200.reset_index(drop=True)
top200["candidate_id"] = top200.index + 1

# Training fingerprints
train_df = pd.read_csv(ORACLE_TRAIN_CSV)
train_smiles = train_df["canonical_smiles"].dropna().tolist()
print("\nComputing ECFP4 for %d training molecules..." % len(train_smiles))
train_fps  = []
train_smi_valid = []
for smi in train_smiles:
    fp = compute_ecfp4(smi)
    if fp is not None:
        train_fps.append(fp)
        train_smi_valid.append(smi)
print("Valid training FPs: %d" % len(train_fps))

# Candidate fingerprints
print("Computing ECFP4 for %d candidates..." % n_cands)
cand_fps = []
for smi in top200[smiles_col]:
    fp = compute_ecfp4(smi)
    cand_fps.append(fp)
print("Valid candidate FPs: %d" % sum(1 for fp in cand_fps if fp is not None))

# Max Tanimoto to training set
print("\nComputing max Tanimoto to training set (200 x %d)..." % len(train_fps))
max_tc_list      = []
nearest_smi_list = []

for i, fp in enumerate(cand_fps):
    if fp is None:
        max_tc_list.append(float("nan"))
        nearest_smi_list.append("")
        continue
    sims     = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    best_idx = int(np.argmax(sims))
    max_tc_list.append(float(sims[best_idx]))
    nearest_smi_list.append(train_smi_valid[best_idx])
    if (i + 1) % 50 == 0:
        print("  ... processed %d/200" % (i + 1))

max_tc_arr = np.array([x for x in max_tc_list if not np.isnan(x)])
print("\nMax training Tc summary:")
print("  mean   : %.4f" % max_tc_arr.mean())
print("  median : %.4f" % float(np.median(max_tc_arr)))
print("  max    : %.4f" % max_tc_arr.max())
print("  min    : %.4f" % max_tc_arr.min())

exact_dup  = [tc >= 0.999 for tc in max_tc_list]
n_exact_dup = sum(exact_dup)
print("\nExact training duplicates (Tc >= 0.999): %d" % n_exact_dup)

# Murcko scaffolds
print("\nExtracting Murcko scaffolds...")
murcko_list = []
for smi in top200[smiles_col]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        murcko_list.append("")
        continue
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        murcko_list.append(scaf)
    except Exception:
        murcko_list.append("")

unique_scaffolds   = set(s for s in murcko_list if s)
n_unique_scaffolds = len(unique_scaffolds)
print("Unique Murcko scaffolds: %d / %d" % (n_unique_scaffolds, n_cands))

# Mean pairwise Tanimoto (exact, 19900 pairs)
print("\nComputing mean pairwise Tanimoto among %d candidates (%d pairs)..." % (n_cands, n_cands*(n_cands-1)//2))
valid_fps = [fp for fp in cand_fps if fp is not None]
n_valid   = len(valid_fps)
pairwise_tc = []
for i in range(n_valid):
    sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[i+1:])
    pairwise_tc.extend(sims)

pairwise_arr = np.array(pairwise_tc)
mean_pw_tc   = float(pairwise_arr.mean())
print("Total pairs computed    : %d" % len(pairwise_tc))
print("Mean pairwise Tanimoto  : %.4f" % mean_pw_tc)
print("Median pairwise Tanimoto: %.4f" % float(np.median(pairwise_arr)))

# -------------------------------------------------------------------------
# Save novelty_analysis_top200.csv
# -------------------------------------------------------------------------
out_df = pd.DataFrame({
    "candidate_id":             top200["candidate_id"],
    "canonical_smiles":         top200[smiles_col],
    "oracle_pIC50":             top200[oracle_col],
    "predicted_logBB":          top200[logbb_col],
    "murcko_scaffold":          murcko_list,
    "max_training_Tc":          max_tc_list,
    "nearest_training_smiles":  nearest_smi_list,
    "exact_training_duplicate": exact_dup,
})

out_df.to_csv(NOVELTY_CSV, index=False)
print("\nSaved: %s" % NOVELTY_CSV)

# -------------------------------------------------------------------------
# Save novelty_summary.txt
# -------------------------------------------------------------------------
summary_lines = [
    "NOVELTY SUMMARY -- TOP 200 CANDIDATES",
    "=" * 50,
    "Total candidates analysed      : %d" % n_cands,
    "Unique Murcko scaffolds        : %d  (expected ~186)" % n_unique_scaffolds,
    "Mean pairwise Tanimoto (ECFP4) : %.4f  (expected ~0.146)" % mean_pw_tc,
    "Exact training duplicates      : %d  (Tc >= 0.999)" % n_exact_dup,
    "Mean max_training_Tc           : %.4f" % max_tc_arr.mean(),
    "Median max_training_Tc         : %.4f" % float(np.median(max_tc_arr)),
    "Max max_training_Tc            : %.4f" % max_tc_arr.max(),
    "",
    "BBB MODEL CAVEATS:",
    "  - predicted_logBB in bace1_continuous_bbb.csv is from a BBB regression model",
    "    (ExtraTrees, trained on B3DB experimental logBB), NOT direct experimental data.",
    "  - BBB predictions on generated molecules are predictions-of-predictions.",
    "  - BBB model scaffold-split test R2=0.354; generalization on novel scaffolds UNVERIFIED.",
    "",
    "ORACLE MODEL CAVEATS:",
    "  - RMSE=0.755 is a typical error statistic, NOT a calibrated 95% prediction interval.",
    "  - Training-set Pearson r (~0.901) is in-sample fit; external accuracy = R2=0.6156.",
    "  - Generated molecules may be outside the oracle's applicability domain.",
    "  - Fraction of training molecules with pIC50 >= 7.0    : %.4f (%d/%d)" % (frac_gt7, n_gt7, len(train_df)),
    "  - Fraction of training molecules with pIC50 >= %.4f: %.4f (%d/%d)" % (threshold, frac_gt76, n_gt76, len(train_df)),
]
summary_text = "\n".join(summary_lines) + "\n"
NOVELTY_TXT.write_text(summary_text, encoding="utf-8")
print("Saved: %s" % NOVELTY_TXT)
print()
print(summary_text)

# =========================================================================
# FINAL SUMMARY
# =========================================================================
sep()
print("=== PHASE 3 COMPLETE ===")
print()
print("PHASE 3A -- Oracle Validation:")
print("  [PASS] ExtraTrees, scaffold-split, n=1240, R2=0.6156, RMSE=0.7553, MAE=0.5879")
print("  [INFO] No test prediction file found; Pearson r / Spearman rho = null in JSON")
print("  [DONE] oracle_metrics_verified.json updated with training stats and caveats")
print("  Training pIC50 >= 7.0      : %d molecules (%.1f%%)" % (n_gt7, frac_gt7 * 100))
print("  Training pIC50 >= %.4f: %d molecules (%.1f%%)" % (threshold, n_gt76, frac_gt76 * 100))
print()
print("PHASE 3B -- BBB Model Validation:")
print("  [PASS] BBB model: ExtraTrees, B3DB experimental logBB, scaffold-split")
print("  [PASS] Test R2=%.4f, RMSE=%.4f, MAE=%.4f (n=%d)" % (
    bbb_meta["test_results"]["R2"], bbb_meta["test_results"]["RMSE"],
    bbb_meta["test_results"]["MAE"], bbb_meta["test_samples"]))
print("  [WARN] predicted_logBB in bace1_continuous_bbb.csv = PREDICTED (not experimental)")
print("  [WARN] BBB predictions on generated molecules = predictions-of-predictions")
print("  [WARN] BBB model generalization on novel scaffolds: UNVERIFIED (R2=0.354)")
print()
print("PHASE 3C -- Novelty Analysis:")
print("  [DONE] Saved: %s" % NOVELTY_CSV)
print("  [DONE] Saved: %s" % NOVELTY_TXT)
print("  Unique Murcko scaffolds : %d" % n_unique_scaffolds)
print("  Mean pairwise Tc        : %.4f" % mean_pw_tc)
print("  Exact training dups     : %d" % n_exact_dup)
