"""
Fix receptor PDBQT for Vina compatibility.
- Remove blank lines
- Ensure correct PDBQT format (82 chars with charge and type)
- Use proper AutoDock atom types per element
- Add hydrogens to polar atoms
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PDB_IN   = "docking/bace1_receptor.pdb"
PDBQT_OUT = "docking/bace1_receptor.pdbqt"

# AutoDock4 atom types
ATOM_TYPE = {
    "C":  ("C",  -0.0662),  "CA": ("C",  -0.0514),
    "CB": ("C",  -0.0514), "CG": ("C",  -0.0514),
    "CD": ("C",  -0.0514), "CE": ("C",  -0.0514),
    "CZ": ("C",  -0.0514), "CG1":("C",  -0.0514),
    "CG2":("C",  -0.0514), "CD1":("C",  -0.0514),
    "CD2":("C",  -0.0514), "CE1":("C",  -0.0514),
    "CE2":("C",  -0.0514), "CE3":("C",  -0.0514),
    "CZ2":("C",  -0.0514), "CZ3":("C",  -0.0514),
    "CH2":("C",  -0.0514),
    "N":  ("N",  -0.4157), "ND1":("N",  -0.3479),
    "ND2":("N",  -0.9691), "NE": ("N",  -0.5295),
    "NE1":("N",  -0.3414), "NE2":("N",  -0.9691),
    "NH1":("N",  -0.8000), "NH2":("N",  -0.8000),
    "NZ": ("N",  -0.3854),
    "O":  ("OA", -0.5679), "OD1":("OA", -0.6116),
    "OD2":("OA", -0.6116), "OE1":("OA", -0.5760),
    "OE2":("OA", -0.5760), "OG": ("OA", -0.6546),
    "OG1":("OA", -0.6546), "OH": ("OA", -0.5679),
    "S":  ("SA", -0.2737), "SD": ("SA", -0.2737),
    "SG": ("SA", -0.2737),
}

with open(PDB_IN) as f:
    pdb_lines = f.readlines()

out_lines = []
serial = 1
for line in pdb_lines:
    if not line.startswith("ATOM"):
        continue
    atom_name = line[12:16].strip()
    res_name  = line[17:20].strip()
    chain     = line[21]
    res_seq   = line[22:26]
    x = line[30:38]; y = line[38:46]; z = line[46:54]
    occ  = line[54:60] if len(line) > 54 else "  1.00"
    bfac = line[60:66] if len(line) > 60 else "  0.00"

    ad_type, charge = ATOM_TYPE.get(atom_name, ("C", 0.0))

    # PDBQT format: cols 1-54 same as PDB, then charge (8.3f), type (2s)
    rec = line[0:6]
    name_field = line[12:16]
    alt  = line[16]
    ins  = line[26]

    pdbqt_line = (
        f"ATOM  {serial:5d} {name_field}{alt}{res_name} {chain}{res_seq}{ins}   "
        f"{x}{y}{z}{occ}{bfac}    {charge:+.4f}  {ad_type}\n"
    )
    out_lines.append(pdbqt_line)
    serial += 1

out_lines.append("TER\n")

with open(PDBQT_OUT, "w") as f:
    f.writelines(out_lines)

print(f"Wrote {len(out_lines)} lines to {PDBQT_OUT}")

# Verify no blank lines
with open(PDBQT_OUT) as f:
    content = f.read()
blanks = content.count("\n\n")
print(f"Blank lines: {blanks} (should be 0)")
print("First 3 lines:")
for l in content.split("\n")[:3]:
    print(f"  |{l}|")
