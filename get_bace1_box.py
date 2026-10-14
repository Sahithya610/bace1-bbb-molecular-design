"""
Extract BACE1 active site center from 2B8V crystal ligand coordinates.
The co-crystallised inhibitor (HETATM OM0) defines the binding site center.
"""
import json
PDB_FILE = "docking/2B8V.pdb"

hetatm = []
with open(PDB_FILE) as f:
    for line in f:
        if line.startswith("HETATM"):
            chain = line[21]
            res = line[17:20].strip()
            if chain == "A" and res not in ("HOH", "SO4", "PO4", "GOL", "EDO", "ACT"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    hetatm.append((res, x, y, z))
                except ValueError:
                    pass

if not hetatm:
    print("No ligand HETATM found in chain A")
else:
    # Group by residue name
    res_names = set(r[0] for r in hetatm)
    print(f"Ligand residues found: {res_names}")

    import numpy as np
    coords = np.array([(x,y,z) for _,x,y,z in hetatm])
    cx, cy, cz = coords.mean(axis=0)
    print(f"Ligand centroid: ({cx:.2f}, {cy:.2f}, {cz:.2f})")

    # Box: 22Å around ligand centroid covers most BACE1 active sites
    box = {
        "center_x": round(cx, 2),
        "center_y": round(cy, 2),
        "center_z": round(cz, 2),
        "size_x": 22.0,
        "size_y": 22.0,
        "size_z": 22.0,
    }
    with open("docking/bace1_box.json", "w") as f:
        json.dump(box, f, indent=2)
    print(f"Updated docking box: {box}")
