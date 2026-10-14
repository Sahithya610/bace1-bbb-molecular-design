"""
Dock top DrugGEN BACE1 candidates using AutoDock-Vina v1.2.7 executable.

Pipeline:
  SMILES → RDKit 3D → PDBQT (meeko) → vina.exe → parse score
"""
import os, sys, json, subprocess, tempfile
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

os.chdir(os.path.dirname(os.path.abspath(__file__)))

VINA_EXE   = "docking/bin/vina.exe"
REC_PDBQT  = "docking/bace1_receptor.pdbqt"
BOX_JSON   = "docking/bace1_box.json"
INPUT_CSV  = "results/top200_scaffold_analysis.csv"
OUTPUT_CSV = "results/docking/bace1_druggen_top_vina.csv"
TOP_N      = 50          # dock top-50 Ro5-pass by oracle pIC50
EXHAUSTIVENESS = 8

print("="*62)
print("BACE1 DOCKING  (AutoDock-Vina v1.2.7)")
print("="*62)

for f in [VINA_EXE, REC_PDBQT, BOX_JSON]:
    if not os.path.exists(f):
        print(f"Missing: {f} — run setup_docking.py first")
        sys.exit(1)

with open(BOX_JSON) as f:
    box = json.load(f)

from meeko import MoleculePreparation, PDBQTWriterLegacy

def smiles_to_pdbqt_file(smiles, path):
    """Write SMILES as PDBQT to path. Returns True on success."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return False
    AllChem.MMFFOptimizeMolecule(mol)
    try:
        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        if not ok: return False
        with open(path, "w") as f:
            f.write(pdbqt)
        return True
    except Exception:
        return False

def run_vina(lig_pdbqt, out_pdbqt):
    """Run vina, return best score (kcal/mol) or None."""
    cmd = [
        VINA_EXE,
        "--receptor", REC_PDBQT,
        "--ligand",   lig_pdbqt,
        "--out",      out_pdbqt,
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x",   str(box["size_x"]),
        "--size_y",   str(box["size_y"]),
        "--size_z",   str(box["size_z"]),
        "--exhaustiveness", str(EXHAUSTIVENESS),
        "--num_modes", "3",
        "--verbosity", "0",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    # Parse best score from stdout (Vina prints scores to stdout)
    # Also try output file
    best = None
    # Try stdout first
    for line in r.stdout.split("\n"):
        if "VINA RESULT" in line:
            try:
                best = float(line.split(":")[1].split()[0])
                break
            except Exception:
                pass
    # Try output file
    if best is None and os.path.exists(out_pdbqt):
        with open(out_pdbqt) as f:
            for line in f:
                if "VINA RESULT" in line:
                    try:
                        best = float(line.split(":")[1].split()[0])
                        break
                    except Exception:
                        pass
    return best

# ── Load & filter candidates ──────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df = df[df["Ro5"] == 1].sort_values("predicted_pIC50", ascending=False).head(TOP_N).reset_index(drop=True)
print(f"Docking {len(df)} candidates (Ro5 pass, top by oracle pIC50)\n")

os.makedirs("docking/ligands", exist_ok=True)
os.makedirs("docking/poses", exist_ok=True)
os.makedirs("results/docking", exist_ok=True)

results = []
for i, row in df.iterrows():
    smi    = row["canonical_smiles"]
    oracle = row["predicted_pIC50"]
    rank   = i + 1
    lig_f  = f"docking/ligands/lig_{rank:03d}.pdbqt"
    out_f  = f"docking/poses/pose_{rank:03d}.pdbqt"

    ok = smiles_to_pdbqt_file(smi, lig_f)
    if not ok:
        print(f"  [{rank:3d}] EMBED FAIL  oracle={oracle:.3f}  {smi[:45]}...")
        results.append({"rank": rank, "canonical_smiles": smi,
                        "oracle_pIC50": oracle, "vina_score": None,
                        "status": "embed_failed"})
        continue

    score = run_vina(lig_f, out_f)
    if score is None:
        print(f"  [{rank:3d}] DOCK FAIL   oracle={oracle:.3f}  {smi[:45]}...")
        results.append({"rank": rank, "canonical_smiles": smi,
                        "oracle_pIC50": oracle, "vina_score": None,
                        "status": "dock_failed"})
    else:
        print(f"  [{rank:3d}] vina={score:+.2f}  oracle={oracle:.3f}  {smi[:45]}...")
        results.append({"rank": rank, "canonical_smiles": smi,
                        "oracle_pIC50": oracle, "vina_score": score,
                        "MW": row["MW"], "LogP": row["LogP"],
                        "TPSA": row["TPSA"], "HBD": row["HBD"],
                        "predicted_logBB": row["predicted_logBB"],
                        "status": "ok"})

# ── Report ────────────────────────────────────────────────────────
res = pd.DataFrame(results)
res.to_csv(OUTPUT_CSV, index=False)

docked = res[res["vina_score"].notna()].sort_values("vina_score")

print(f"\n{'='*62}")
print(f"RESULTS  —  {len(docked)}/{len(df)} docked successfully")
print(f"{'='*62}")
if len(docked):
    print(f"Best Vina:  {docked['vina_score'].min():.2f} kcal/mol")
    print(f"Mean Vina:  {docked['vina_score'].mean():.2f} kcal/mol")
    print(f"\n  {'#':>3}  {'Vina':>6}  {'Oracle':>7}  {'logBB':>6}  {'MW':>6}  SMILES")
    print("  " + "-"*85)
    for j, (_, r) in enumerate(docked.head(15).iterrows(), 1):
        s = r["canonical_smiles"]
        s = s[:52] + "..." if len(s) > 52 else s
        print(f"  {j:3d}  {r['vina_score']:+6.2f}  "
              f"{r['oracle_pIC50']:7.3f}  "
              f"{r['predicted_logBB']:+6.3f}  "
              f"{r['MW']:6.1f}  {s}")

print(f"\nSaved: {OUTPUT_CSV}")
