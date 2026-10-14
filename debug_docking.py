"""Debug: run vina on one ligand, print full output."""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy

smi = "CC(C)(C)COc1cnc2c(c1)C1(COC(N)=N1)c1cc(-c3cncnc3)ccc1O2"
mol = Chem.AddHs(Chem.MolFromSmiles(smi))
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol)

prep = MoleculePreparation()
setups = prep.prepare(mol)
pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
print(f"Ligand PDBQT ok={ok}")
if not ok:
    print(f"Error: {err}")
    sys.exit(1)

with open("docking/debug_lig.pdbqt","w") as f: f.write(pdbqt)

# Run vina verbosely
r = subprocess.run([
    "docking/bin/vina.exe",
    "--receptor", "docking/bace1_receptor.pdbqt",
    "--ligand",   "docking/debug_lig.pdbqt",
    "--out",      "docking/debug_out.pdbqt",
    "--center_x","4.0","--center_y","22.0","--center_z","18.0",
    "--size_x","20","--size_y","20","--size_z","20",
    "--exhaustiveness","4","--num_modes","1","--verbosity","2",
], capture_output=True, text=True)

print("STDOUT:", r.stdout[:2000])
print("STDERR:", r.stderr[:2000])
print("Return code:", r.returncode)

# Check output
if os.path.exists("docking/debug_out.pdbqt"):
    with open("docking/debug_out.pdbqt") as f:
        content = f.read()
    print("Output PDBQT:", content[:500])
else:
    print("No output file created")
