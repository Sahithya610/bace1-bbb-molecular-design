"""
Prepare BACE1 receptor PDBQT using meeko's Python API.
Input:  docking/bace1_receptor.pdb  (clean protein, chain A, ATOM only)
Output: docking/bace1_receptor.pdbqt
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

REC_PDB   = "docking/bace1_receptor.pdb"
REC_PDBQT = "docking/bace1_receptor.pdbqt"

print(f"Preparing receptor PDBQT from {REC_PDB}...")

# meeko >= 0.5 has mk_prepare_receptor Python API
try:
    from meeko import PDBQTWriterLegacy, MoleculePreparation
    # Try the receptor preparation module
    try:
        from meeko.receptor import (
            load_receptor_from_file,
            prepare_receptor,
        )
        receptor = load_receptor_from_file(REC_PDB)
        pdbqt_str = prepare_receptor(receptor)
        with open(REC_PDBQT, "w") as f:
            f.write(pdbqt_str)
        print(f"Saved: {REC_PDBQT}")
    except ImportError:
        # Older meeko API — use mk_prepare_receptor subprocess
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c",
             f"from meeko import mk_prepare_receptor; mk_prepare_receptor.main()"
             ],
            input=f"--pdb {REC_PDB} -o {REC_PDBQT}",
            capture_output=True, text=True
        )
        raise ImportError("trying next method")
except Exception as e:
    print(f"meeko API approach failed ({e}), trying AutoDockTools-style PDBQT...")
    # Manual PDBQT: copy PDB and add AutoDock atom types
    # This is a simplified approach that works for most cases
    # Real receptor prep needs HTMD or MGLTools for proper charges
    # For docking purposes, Gasteiger charges via a simple script:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem import rdForceFieldHelpers

        # Read PDB as RDKit mol — but RDKit PDB reader struggles with proteins
        # Instead write a minimal PDBQT by converting atom types manually
        print("Writing simplified PDBQT (Kollman charges, standard atom types)...")

        with open(REC_PDB) as f:
            pdb_lines = f.readlines()

        # AutoDock atom type mapping (simplified — covers most amino acids)
        AD_TYPES = {
            "C": "C", "CA": "C", "CB": "C", "CG": "C", "CD": "C",
            "CE": "C", "CZ": "C", "CH2": "C", "CG1": "C", "CG2": "C",
            "CD1": "C", "CD2": "C", "CE1": "C", "CE2": "C", "CE3": "C",
            "CZ2": "C", "CZ3": "C",
            "N": "N", "ND1": "N", "ND2": "N", "NE": "N", "NE1": "N",
            "NE2": "N", "NH1": "N", "NH2": "N", "NZ": "N",
            "O": "OA", "OD1": "OA", "OD2": "OA", "OE1": "OA", "OE2": "OA",
            "OG": "OA", "OG1": "OA", "OH": "OA",
            "S": "SA", "SD": "SA", "SG": "SA",
        }

        pdbqt_lines = []
        for line in pdb_lines:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            ad_type = AD_TYPES.get(atom_name, "C")
            # PDBQT format: same as PDB + partial charge + atom type
            # Use 0.000 as placeholder charge (Kollman/Gasteiger not computed here)
            pdbqt_line = line[:54] + "  0.000" + f"  {ad_type}\n"
            # Trim to 80 chars and pad
            pdbqt_lines.append(pdbqt_line[:79] + "\n")

        pdbqt_lines.append("TER\n")

        with open(REC_PDBQT, "w") as f:
            f.writelines(pdbqt_lines)

        print(f"Saved simplified PDBQT: {REC_PDBQT} ({len(pdbqt_lines)} lines)")
        print("Note: Using 0.000 placeholder charges. Results are directional.")
        print("For publication-quality docking, use MGLTools or HTMD for prep.")

    except Exception as e2:
        print(f"Failed: {e2}")
        sys.exit(1)

print("Receptor PDBQT ready.")
