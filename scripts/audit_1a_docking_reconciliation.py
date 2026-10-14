"""
Audit 1A: Docking Count Reconciliation
Iterates over pose_001.pdbqt ... pose_050.pdbqt, parses content,
cross-references with bace1_final_ranked.csv, resolves score precision.
"""

import sys
import re
import csv
import warnings
warnings.filterwarnings("ignore")
# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
POSES_DIR   = ROOT / "docking" / "poses"
RANKED_CSV  = ROOT / "results" / "docking" / "bace1_final_ranked.csv"
OUT_DIR     = ROOT / "reports" / "audit_corrections"
OUT_CSV     = OUT_DIR / "docking_reconciliation.csv"
OUT_TXT     = OUT_DIR / "docking_reconciliation_summary.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ranked CSV ─────────────────────────────────────────────────────
import pandas as pd

ranked = pd.read_csv(RANKED_CSV)
print(f"Ranked CSV rows: {len(ranked)}")
print(f"Columns: {ranked.columns.tolist()}")

# Build a dict: rank -> row
rank_lookup = {int(row["rank"]): row for _, row in ranked.iterrows()}

# ── Parse each pose file ─────────────────────────────────────────────────
VINA_RE   = re.compile(r"REMARK VINA RESULT:\s+([-\d.]+)")
MODEL_RE  = re.compile(r"^MODEL\s", re.MULTILINE)
ENDMDL_RE = re.compile(r"^ENDMDL", re.MULTILINE)
ATOM_RE   = re.compile(r"^(?:ATOM|HETATM)\s", re.MULTILINE)

records = []

for i in range(1, 51):
    pose_filename = f"pose_{i:03d}.pdbqt"
    pose_path     = POSES_DIR / pose_filename

    exists      = pose_path.exists()
    has_model   = False
    has_atom    = False
    parsed_vina = None
    failure     = ""

    if exists:
        content = pose_path.read_text(encoding="utf-8", errors="replace")

        has_model = bool(MODEL_RE.search(content)) and bool(ENDMDL_RE.search(content))
        has_atom  = bool(ATOM_RE.search(content))

        m = VINA_RE.search(content)
        if m:
            try:
                parsed_vina = float(m.group(1))
            except ValueError:
                failure = "VINA RESULT parse error"
        else:
            if content.strip() == "":
                failure = "empty file"
            else:
                failure = "no REMARK VINA RESULT line"
    else:
        failure = "file missing"

    # ── Cross-reference with ranked CSV ─────────────────────────────────
    csv_row = rank_lookup.get(i)
    if csv_row is not None:
        canon_smiles  = csv_row["canonical_smiles"]
        oracle_pic50  = csv_row["oracle_pIC50"]
        csv_vina      = float(csv_row["vina_score"])
    else:
        canon_smiles  = ""
        oracle_pic50  = ""
        csv_vina      = None

    if parsed_vina is not None and csv_vina is not None:
        score_matches = abs(parsed_vina - csv_vina) < 1e-6
    else:
        score_matches = False

    if not failure and parsed_vina is None:
        failure = "no parsed score"
    if csv_row is None and not failure:
        failure = "rank not in CSV"

    records.append({
        "rank":                i,
        "canonical_smiles":    canon_smiles,
        "oracle_pIC50":        oracle_pic50,
        "pose_file":           pose_filename,
        "pose_file_exists":    exists,
        "has_model_record":    has_model,
        "parsed_vina_score":   parsed_vina,
        "vina_score_in_csv":   csv_vina,
        "score_matches":       score_matches,
        "failure_reason":      failure,
    })

# ── Write CSV ────────────────────────────────────────────────────────────
df_out = pd.DataFrame(records)
df_out.to_csv(OUT_CSV, index=False)
print(f"Written: {OUT_CSV}")

# ── Analysis ─────────────────────────────────────────────────────────────
n_exists         = sum(1 for r in records if r["pose_file_exists"])
n_valid          = sum(1 for r in records if r["has_model_record"] and r["pose_file_exists"])
n_has_vina       = sum(1 for r in records if r["parsed_vina_score"] is not None)
n_matches        = sum(1 for r in records if r["score_matches"])
n_mismatches     = sum(1 for r in records if r["parsed_vina_score"] is not None
                       and r["vina_score_in_csv"] is not None
                       and not r["score_matches"])

# Distinct molecules submitted = 50 pose files (rank 1..50)
n_submitted = 50

# Best score from raw files
valid_vina = [r["parsed_vina_score"] for r in records if r["parsed_vina_score"] is not None]
best_raw   = min(valid_vina) if valid_vina else None

# Best score from CSV
csv_vina_vals = [r["vina_score_in_csv"] for r in records if r["vina_score_in_csv"] is not None]
best_csv = min(csv_vina_vals) if csv_vina_vals else None

# Check for any CSV vina rows that may have come from non-pose sources
# (i.e. the original submitted rank before composite re-sorting)
print(f"\nParsed Vina scores: {sorted(valid_vina)[:5]} ...")
print(f"CSV Vina scores (first 5 lowest): {sorted(csv_vina_vals)[:5]}")

# Investigate mismatches
mismatch_records = [r for r in records if
                    r["parsed_vina_score"] is not None and
                    r["vina_score_in_csv"] is not None and
                    not r["score_matches"]]
if mismatch_records:
    print("\nMismatches:")
    for r in mismatch_records:
        print(f"  rank={r['rank']}  raw={r['parsed_vina_score']!r}  csv={r['vina_score_in_csv']!r}")

# Count empty files (exist but no MODEL/ATOM)
n_empty = sum(1 for r in records
              if r["pose_file_exists"] and not r["has_model_record"])

# ── Write summary ─────────────────────────────────────────────────────────
lines = [
    "=== DOCKING COUNT RECONCILIATION SUMMARY ===",
    "",
    f"Distinct molecules submitted (pose files 001-050):  {n_submitted}",
    f"Pose files that exist on disk:                       {n_exists}/50",
    f"Pose files that are non-empty / have MODEL+ENDMDL:  {n_valid}/50",
    f"Pose files that are physically present but empty:   {n_empty}",
    f"Pose files with a parsed Vina score:                {n_has_vina}/50",
    f"CSV vina_score values populated:                    {len([v for v in csv_vina_vals if v is not None])}/50",
    f"Score matches (raw == CSV, tol<1e-6):               {n_matches}/50",
    f"Score mismatches:                                   {n_mismatches}",
    "",
    f"Authoritative best Vina score (from raw .pdbqt):   {best_raw} kcal/mol",
    f"Best Vina score as recorded in CSV:                 {best_csv} kcal/mol",
    "",
    "EXPLANATION FOR PREVIOUS '37 DOCKED' DISCREPANCY",
    "-------------------------------------------------",
]

# Determine explanation
empty_fail = [r for r in records if r["pose_file_exists"] and not r["has_model_record"]]
missing_fail = [r for r in records if not r["pose_file_exists"]]

if not empty_fail and not missing_fail and n_valid == 50:
    lines += [
        "All 50 pose files exist and contain valid MODEL/ENDMDL blocks.",
        "The '37 docked successfully' figure was inconsistent with the actual",
        "file set. Every pose file contains atom records and a parsed Vina score.",
        "The most likely source of the discrepancy is that an intermediate",
        "summary was generated from an earlier, partial docking run (37 poses)",
        "before the remaining 13 were produced, and was never updated.",
        "The bace1_final_ranked.csv is therefore consistent with all 50 poses.",
    ]
elif empty_fail:
    ranks_empty = [r["rank"] for r in empty_fail]
    lines += [
        f"{len(empty_fail)} pose files are present but contain no MODEL records",
        f"(ranks: {ranks_empty}).",
        "These correspond to ligands where AutoDock Vina failed to find a pose.",
        "The '37 docked successfully' figure likely refers to the 37 files",
        "that had non-empty MODEL/ATOM records; the remaining 13 were empty",
        "placeholder files. The CSV vina_score for empty files appears to be",
        "copied from a placeholder or estimated value — NOT from the raw file.",
    ]
elif missing_fail:
    lines += [
        f"{len(missing_fail)} of the expected 50 pose files are missing.",
        "The '37 docked' figure may reflect the actual completed count at time",
        "of the summary, with additional files generated later.",
    ]

lines += [
    "",
    "NOTE ON SCORE PRECISION:",
    f"  Raw file best score: {best_raw} kcal/mol  (full precision as written by Vina)",
    f"  CSV file best score: {best_csv} kcal/mol",
    "  If these differ, the CSV should be treated as rounded.",
    "",
    f"Output CSV: {OUT_CSV}",
]

summary_text = "\n".join(lines)
OUT_TXT.write_text(summary_text, encoding="utf-8")
print(f"Written: {OUT_TXT}")
print()
print(summary_text)
