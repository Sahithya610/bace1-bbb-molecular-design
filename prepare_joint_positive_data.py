"""
Phase 1 Data Preparation: Joint-Positive Target-Specific DrugGEN Training
Audits BACE1 molecules meeting BOTH pIC50 >= 7.6576 AND predicted_logBB >= 0.0,
performs graph-matching check, scaffold-aware split, and writes training/held-out files.
"""
import os
import sys
import json
import pathlib

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---------------------------------------------------------------------------
# Paths
ROOT = pathlib.Path(__file__).resolve().parent
CSV_PATH = ROOT / "druggen_data" / "bace1_continuous_bbb.csv"
REPORTS_DIR = ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "joint_positive_data_prep.json"
TRAIN_SMI = ROOT / "druggen_data" / "bace1_joint_positive_train.smi"
HELDOUT_SMI = ROOT / "druggen_data" / "bace1_joint_positive_heldout.smi"

PIC50_THRESHOLD = 7.6576
LOGBB_THRESHOLD = 0.0
MIN_FOR_EXPERIMENT = 10
MIN_FOR_SPLIT = 20

GRAPH_CANDIDATES = [
    ROOT / "druggen_data" / "bace145.pt",
    ROOT / "data" / "bace145.pt",
]

# ---------------------------------------------------------------------------
# Helper: canonicalize SMILES
# ---------------------------------------------------------------------------
def canonicalize(smi):
    """Return canonical SMILES or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def get_murcko(smi):
    """Return Murcko scaffold SMILES or empty string for ring-free molecules."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Step 1a: Audit
# ---------------------------------------------------------------------------
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 1a: Auditing joint-positive BACE1 molecules")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
total_in_csv = len(df)
print(f"Total records in CSV: {total_in_csv}")

# Count threshold hits
n_pic50 = int((df["pIC50"] >= PIC50_THRESHOLD).sum())
n_logBB = int((df["predicted_logBB"] >= LOGBB_THRESHOLD).sum())
joint_mask = (df["pIC50"] >= PIC50_THRESHOLD) & (df["predicted_logBB"] >= LOGBB_THRESHOLD)
n_joint_raw = int(joint_mask.sum())

print(f"pIC50 >= {PIC50_THRESHOLD}: {n_pic50}")
print(f"predicted_logBB >= {LOGBB_THRESHOLD}: {n_logBB}")
print(f"Joint positive (raw rows): {n_joint_raw}")

# Canonicalize joint-positive SMILES
joint_df = df[joint_mask].copy()
joint_df["canon_smi"] = joint_df["canonical_smiles"].apply(canonicalize)
joint_df = joint_df.dropna(subset=["canon_smi"])

# Count duplicates in original CSV (before canonicalization)
n_duplicates_in_csv = int(joint_df.duplicated(subset=["canonical_smiles"]).sum())

# Unique canonical SMILES
unique_canon = joint_df.drop_duplicates(subset=["canon_smi"])
n_joint_unique = len(unique_canon)
print(f"Unique canonical SMILES among joint-positive: {n_joint_unique}")
print(f"Duplicate SMILES in joint-positive CSV rows: {n_duplicates_in_csv}")

# Label sources
print()
print("Label sources:")
print("  pIC50: ChEMBL experimental IC50, formula 9-log10(IC50_nM)")
print("  predicted_logBB: B3DB ExtraTrees model predictions, test R2=0.354, NOT experimental")

# ---------------------------------------------------------------------------
# Step 1b: Graph matching
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 1b: Graph matching check")
print("=" * 60)

graph_file_used = None
n_joint_with_graph = None

# Find graph file
for candidate in GRAPH_CANDIDATES:
    if candidate.exists():
        graph_file_used = str(candidate)
        break

if graph_file_used is None:
    # Search for any .pt file with 'bace' in the name
    for search_dir in [ROOT / "druggen_data", ROOT / "data"]:
        if search_dir.exists():
            for pt_file in search_dir.glob("*.pt"):
                if "bace" in pt_file.name.lower():
                    graph_file_used = str(pt_file)
                    break
        if graph_file_used:
            break

if graph_file_used:
    print(f"Graph file found: {graph_file_used}")
    try:
        import torch
        graph_data = torch.load(graph_file_used, map_location="cpu", weights_only=False)

        # Extract SMILES from graph dataset.
        # The .pt file is a tuple of torch_geometric Data objects.
        # Each Data object's .smiles attribute may be a list of all dataset SMILES
        # (stored once on the first element) or a single string per molecule.
        graph_smiles_set = set()

        items = graph_data if isinstance(graph_data, (list, tuple)) else [graph_data]

        for item in items:
            # Handle .smiles that is a list (batch-style storage)
            smi_val = None
            if hasattr(item, "smiles"):
                smi_val = item.smiles
            elif hasattr(item, "smi"):
                smi_val = item.smi
            elif isinstance(item, dict) and "smiles" in item:
                smi_val = item["smiles"]

            if smi_val is None:
                continue

            if isinstance(smi_val, (list, tuple)):
                # All SMILES stored as a list on one element
                for s in smi_val:
                    canon = canonicalize(str(s))
                    if canon:
                        graph_smiles_set.add(canon)
            else:
                canon = canonicalize(str(smi_val))
                if canon:
                    graph_smiles_set.add(canon)

        # Also try indexing if dataset has __len__ and __getitem__
        if not graph_smiles_set and hasattr(graph_data, "__len__") and not isinstance(graph_data, (list, tuple)):
            for i in range(len(graph_data)):
                try:
                    item = graph_data[i]
                    smi_val = None
                    if hasattr(item, "smiles"):
                        smi_val = item.smiles
                    elif hasattr(item, "smi"):
                        smi_val = item.smi
                    if smi_val is not None:
                        if isinstance(smi_val, (list, tuple)):
                            for s in smi_val:
                                c = canonicalize(str(s))
                                if c:
                                    graph_smiles_set.add(c)
                        else:
                            c = canonicalize(str(smi_val))
                            if c:
                                graph_smiles_set.add(c)
                except Exception:
                    pass

        print(f"Graph dataset SMILES extracted: {len(graph_smiles_set)}")

        if graph_smiles_set:
            joint_canon_set = set(unique_canon["canon_smi"].tolist())
            matched = joint_canon_set & graph_smiles_set
            n_joint_with_graph = len(matched)
            print(f"Joint-positive SMILES found in graph: {n_joint_with_graph}")
        else:
            print("  WARNING: Could not extract SMILES from graph file; using CSV SMILES as proxy.")
            n_joint_with_graph = n_joint_unique
            graph_file_used = graph_file_used + " (SMILES not extractable; using CSV as proxy)"

    except Exception as e:
        print(f"  ERROR loading graph file: {e}")
        print("  Using CSV SMILES as proxy.")
        n_joint_with_graph = n_joint_unique
        graph_file_used = graph_file_used + " (load error; using CSV as proxy)"
else:
    print("No graph file found. Using CSV canonical SMILES as proxy.")
    n_joint_with_graph = n_joint_unique

print(f"Joint-positive with graph match: {n_joint_with_graph} / {n_joint_unique}")

# ---------------------------------------------------------------------------
# Stopping rule
# ---------------------------------------------------------------------------
effective_count = n_joint_with_graph if n_joint_with_graph is not None else n_joint_unique
if effective_count < MIN_FOR_EXPERIMENT:
    report = {
        "status": "STOPPED",
        "reason": "insufficient joint-positive samples",
        "n_joint_positive": effective_count,
        "threshold": MIN_FOR_EXPERIMENT,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print()
    print("STOPPING: Insufficient joint-positive samples for this approach.")
    print(f"  n_joint_positive={effective_count} < threshold={MIN_FOR_EXPERIMENT}")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Step 1c: Scaffold-aware split
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 1c: Scaffold-aware split")
print("=" * 60)

working_smiles = unique_canon["canon_smi"].tolist()
n_available = len(working_smiles)
print(f"Available unique canonical SMILES: {n_available}")

train_smiles = []
heldout_smiles = []
split_strategy = ""
scaffold_overlap = 0
exact_overlap = 0

if n_available >= MIN_FOR_SPLIT:
    # Compute scaffolds
    scaffold_to_smiles = {}
    for smi in working_smiles:
        scaf = get_murcko(smi)
        scaffold_to_smiles.setdefault(scaf, []).append(smi)

    # Sort scaffolds by number of members descending
    sorted_scaffolds = sorted(scaffold_to_smiles.keys(), key=lambda s: len(scaffold_to_smiles[s]), reverse=True)

    # Greedy 80/20 split at scaffold level
    target_train = int(round(0.8 * n_available))
    for scaf in sorted_scaffolds:
        members = scaffold_to_smiles[scaf]
        if len(train_smiles) < target_train:
            train_smiles.extend(members)
        else:
            heldout_smiles.extend(members)

    # Verify scaffold overlap (should be 0)
    train_scaffolds = set(get_murcko(s) for s in train_smiles)
    held_scaffolds = set(get_murcko(s) for s in heldout_smiles)
    scaffold_overlap = len(train_scaffolds & held_scaffolds)
    exact_overlap = len(set(train_smiles) & set(heldout_smiles))

    split_strategy = "scaffold_80_20"
    print(f"Scaffold split: {len(train_smiles)} train / {len(heldout_smiles)} held-out")
    print(f"Scaffold overlap between train and held-out: {scaffold_overlap}")
    print(f"Exact SMILES overlap between train and held-out: {exact_overlap}")

else:
    train_smiles = working_smiles
    heldout_smiles = []
    split_strategy = "all_train_insufficient_for_split"
    print(f"n_available={n_available} < {MIN_FOR_SPLIT}: all molecules used for training, no held-out set.")

# ---------------------------------------------------------------------------
# Write output files
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("WRITING OUTPUT FILES")
print("=" * 60)

with open(TRAIN_SMI, "w") as f:
    for smi in train_smiles:
        f.write(smi + "\n")
print(f"Written: {TRAIN_SMI}  ({len(train_smiles)} SMILES)")

with open(HELDOUT_SMI, "w") as f:
    if heldout_smiles:
        for smi in heldout_smiles:
            f.write(smi + "\n")
    else:
        f.write("# No held-out set: n_joint_positive < 20\n")
print(f"Written: {HELDOUT_SMI}  ({len(heldout_smiles)} SMILES)")

# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
report = {
    "status": "OK",
    "total_in_csv": total_in_csv,
    "n_pic50_ge7_6576": n_pic50,
    "n_logBB_ge0": n_logBB,
    "n_joint_positive_raw": n_joint_raw,
    "n_joint_positive_unique_canonical": n_joint_unique,
    "n_joint_positive_with_graph": n_joint_with_graph,
    "graph_file_used": graph_file_used,
    "n_train": len(train_smiles),
    "n_heldout": len(heldout_smiles),
    "split_strategy": split_strategy,
    "scaffold_overlap_train_heldout": scaffold_overlap,
    "exact_overlap_train_heldout": exact_overlap,
    "n_duplicates_in_csv": n_duplicates_in_csv,
    "bbb_label_source": "B3DB ExtraTrees model predictions, test R2=0.354, NOT experimental",
    "pic50_label_source": "ChEMBL experimental IC50, formula 9-log10(IC50_nM)",
    "pic50_threshold": PIC50_THRESHOLD,
    "logBB_threshold": LOGBB_THRESHOLD,
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"Written: {REPORT_PATH}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total CSV records:               {total_in_csv}")
print(f"Joint-positive raw rows:         {n_joint_raw}")
print(f"Joint-positive unique canonical: {n_joint_unique}")
print(f"Graph-matched count:             {n_joint_with_graph}")
print(f"Train set size:                  {len(train_smiles)}")
print(f"Held-out set size:               {len(heldout_smiles)}")
print(f"Split strategy:                  {split_strategy}")
print(f"Scaffold overlap (train/heldout): {scaffold_overlap}")
print()
can_proceed = effective_count >= MIN_FOR_EXPERIMENT
print(f"Experiment can proceed: {can_proceed}")
print("=" * 60)
