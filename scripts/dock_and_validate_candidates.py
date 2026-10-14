"""
Dock and Validate Candidates Across Receptor Protonation States
Executes validated Vina docking on reference 3BN and all candidates.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy

ROOT = Path(__file__).resolve().parent.parent
DOCK_DIR = ROOT / "docking"
BIN_VINA = DOCK_DIR / "bin" / "vina.exe"

candidates = {
    "Candidate_1308": "CC1CCC(COc2ccc(F)cn2)CN1C(O)c1cc(F)ccc1-c1ncccn1",
    "Candidate_1902": "OC1CC2(c3cc(F)ccc3NC2O)c2cnn(Cc3ccc(C(F)(F)F)cc3)c2N1",
    "ANALOGUE_0011":  "CC(C)(C)COc1cnc2c(c1)C1(COC(N)=N1)c1cc(-c3cncnc3F)ccc1O2",
    "Candidate_3":    "CCC1CC2(CCC1OC)Cc1ccc(-c3cncnc3)cc1C21COC(N)=N1",
}

print("=== Step 1: Ligand PDBQT Preparation ===")
for name, smi in candidates.items():
    lig_pdbqt_path = DOCK_DIR / f"{name}.pdbqt"
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    pdbqt_str, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
    with open(lig_pdbqt_path, "w") as f:
        f.write(pdbqt_str)
    print(f"Prepared {lig_pdbqt_path.name}: ok={ok}")

receptors = [
    ("deprot", "bace1_receptor"),
    ("ash32",  "bace1_receptor_ash32"),
    ("ash228", "bace1_receptor_ash228")
]

print("\n=== Step 2: 3BN Redocking and Validation ===")
# Load crystal 3BN coords
crystal_by_name = {}
with open(DOCK_DIR / "ligand_3BN_crystal.pdb") as f:
    for l in f:
        if l.startswith("HETATM") or l.startswith("ATOM"):
            atom_name = l[12:16].strip()
            x = float(l[30:38])
            y = float(l[38:46])
            z = float(l[46:54])
            crystal_by_name[atom_name] = np.array([x, y, z])

redock_results = {}
for code, rec_file in receptors:
    out_file = DOCK_DIR / f"redock_3BN_{code}_out.pdbqt"
    cmd = [
        str(BIN_VINA),
        "--receptor", str(DOCK_DIR / f"{rec_file}.pdbqt"),
        "--ligand",   str(DOCK_DIR / "ligand_3BN_crystal.pdbqt"),
        "--out",      str(out_file),
        "--center_x", "30.65",
        "--center_y", "41.54",
        "--center_z", "1.86",
        "--size_x",   "22.0",
        "--size_y",   "22.0",
        "--size_z",   "22.0",
        "--exhaustiveness", "8",
        "--num_modes", "5"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse pose 1
    pose1_atoms = {}
    score = None
    with open(out_file) as f:
        in_model_1 = False
        for l in f:
            if "REMARK VINA RESULT:" in l and score is None:
                score = float(l.split()[3])
            elif l.startswith("MODEL 1"):
                in_model_1 = True
            elif l.startswith("ENDMDL") and in_model_1:
                break
            elif in_model_1 and (l.startswith("ATOM") or l.startswith("HETATM")):
                aname = l[12:16].strip()
                if not aname.startswith("H") and l[77:79].strip() != "H":
                    pose1_atoms[aname] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    
    diffs = [np.sum((crystal_by_name[k] - pose1_atoms[k])**2) for k in crystal_by_name if k in pose1_atoms]
    rmsd = float(np.sqrt(np.mean(diffs)))
    redock_results[code] = {
        "receptor_variant": rec_file,
        "vina_score": score,
        "heavy_atom_rmsd": round(rmsd, 3),
        "mapped_heavy_atoms": len(diffs),
        "validation_passed": (rmsd < 2.0)
    }
    print(f"3BN Redock [{code}]: Score={score:.2f} kcal/mol, RMSD={rmsd:.3f} A, Passed={rmsd < 2.0}")

print("\n=== Step 3: Docking Candidates Across Protonation States ===")
candidate_scores = {name: {} for name in candidates}

for code, rec_file in receptors:
    for name in candidates:
        out_file = DOCK_DIR / f"{name}_{code}_out.pdbqt"
        cmd = [
            str(BIN_VINA),
            "--receptor", str(DOCK_DIR / f"{rec_file}.pdbqt"),
            "--ligand",   str(DOCK_DIR / f"{name}.pdbqt"),
            "--out",      str(out_file),
            "--center_x", "30.65",
            "--center_y", "41.54",
            "--center_z", "1.86",
            "--size_x",   "22.0",
            "--size_y",   "22.0",
            "--size_z",   "22.0",
            "--exhaustiveness", "8",
            "--num_modes", "3"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        score = None
        if out_file.exists():
            with open(out_file) as f:
                for l in f:
                    if "REMARK VINA RESULT:" in l:
                        score = float(l.split()[3])
                        break
        candidate_scores[name][code] = score
        print(f"Dock [{code}] {name}: {score} kcal/mol")

print("\n=== Summary Table ===")
print("Candidate | Deprot (kcal/mol) | ASH32 (kcal/mol) | ASH228 (kcal/mol)")
for name in candidates:
    print(f"{name:15} | {candidate_scores[name]['deprot']:>8.2f} | {candidate_scores[name]['ash32']:>8.2f} | {candidate_scores[name]['ash228']:>8.2f}")

# Save JSON result
out_json = {
    "redocking_validation": redock_results,
    "candidate_scores": candidate_scores
}
with open(DOCK_DIR / "multi_state_docking_results.json", "w") as f:
    json.dump(out_json, f, indent=2)

print(f"\nSaved docking results to {DOCK_DIR / 'multi_state_docking_results.json'}")
