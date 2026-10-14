"""
inspect_mappings.py  (READ-ONLY: Steps 1-2 of the DrugGEN handoff)

Run from the project root:
    python inspect_mappings.py

It does not write, regenerate, or modify any mapping, checkpoint or result file.
Output is also saved to results/step2_mapping_report.txt (new file only).
"""
import glob
import os
import pickle
import sys
from collections import Counter

ROOT = os.getcwd()
OUT_LINES = []


def log(msg=""):
    print(msg)
    OUT_LINES.append(str(msg))


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    log(f"Project root: {ROOT}")

    # ---- 1. Mapping files -------------------------------------------------
    enc_dir = os.path.join(ROOT, "data", "encoders")
    dec_dir = os.path.join(ROOT, "data", "decoders")
    for d in (enc_dir, dec_dir):
        log(f"\n{d}:")
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            log(f"  {os.path.basename(p)}  ({os.path.getsize(p)} bytes, mtime {os.path.getmtime(p):.0f})")

    atom_decs = sorted(glob.glob(os.path.join(dec_dir, "atom_*.pkl")))
    if not atom_decs:
        log("\nNo atom decoder pickles found. Stop and report this.")
        return

    for atom_dec_path in atom_decs:
        suffix = os.path.basename(atom_dec_path)[len("atom_"):]
        bond_dec_path = os.path.join(dec_dir, "bond_" + suffix)
        atom_enc_path = os.path.join(enc_dir, "atom_" + suffix)
        log("\n" + "=" * 70)
        log(f"Mapping set suffix: {suffix}")
        atom_dec = load_pkl(atom_dec_path)
        log(f"atom_decoder (class -> atomic number), size {len(atom_dec)}:")
        log(f"  {atom_dec}")
        if os.path.exists(atom_enc_path):
            atom_enc = load_pkl(atom_enc_path)
            log(f"atom_encoder (atomic number -> class): {atom_enc}")
            log(f"  padding (Z=0) class: {atom_enc.get(0)}")
            for z, name in [(6, "C"), (7, "N"), (8, "O"), (9, "F"), (16, "S"), (17, "Cl")]:
                log(f"  {name} (Z={z}) class: {atom_enc.get(z, 'ABSENT')}")
        if os.path.exists(bond_dec_path):
            bond_dec = load_pkl(bond_dec_path)
            log(f"bond_decoder (class -> RDKit BondType), size {len(bond_dec)}:")
            log(f"  {bond_dec}")
            zero = [k for k, v in bond_dec.items() if v == Chem.rdchem.BondType.ZERO]
            log(f"  no-bond (BondType.ZERO) class: {zero}")
        else:
            log(f"Missing bond decoder: {bond_dec_path}")

    # ---- 2. Element frequencies in the raw SMILES inputs -------------------
    for smi_path in ("data/chembl_train.smi", "druggen_data/bace1.smi"):
        full = os.path.join(ROOT, smi_path)
        if not os.path.exists(full):
            log(f"\n{smi_path}: NOT FOUND")
            continue
        counts, n_ok, n_bad, n_o = Counter(), 0, 0, 0
        with open(full) as f:
            for i, line in enumerate(f):
                if i >= 200000:  # cap for speed; note it in the report
                    break
                smi = line.strip().split()[0] if line.strip() else ""
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol is None or mol.GetNumAtoms() > 45:
                    n_bad += 1
                    continue
                n_ok += 1
                zs = {a.GetAtomicNum() for a in mol.GetAtoms()}
                n_o += 8 in zs
                counts.update(a.GetAtomicNum() for a in mol.GetAtoms())
        log(f"\n{smi_path} (first 200000 lines max): usable={n_ok}, skipped={n_bad}")
        log(f"  molecules containing oxygen: {n_o} ({100 * n_o / max(n_ok, 1):.2f}%)")
        log(f"  atom counts by atomic number: {dict(sorted(counts.items()))}")

    # ---- 3. Checkpoint tensor shapes (no model construction) ---------------
    import torch
    for ckpt in (
        "experiments/models/Issue3-ChemistryReadout/joint_generator.pt",
        "experiments/models/Issue3-Diagnostic/generator_epoch_002.pt",
    ):
        full = os.path.join(ROOT, ckpt)
        log("\n" + "=" * 70)
        if not os.path.exists(full):
            log(f"{ckpt}: NOT FOUND")
            continue
        sd = torch.load(full, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "generator" in sd:
            sd = sd["generator"]
        log(f"{ckpt}: {len(sd)} tensors")
        for k, v in sd.items():
            if any(t in k for t in ("readout", "condition", "aux", "property")):
                log(f"  {k}: {tuple(v.shape)}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", "step2_mapping_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT_LINES))
    log(f"\nSaved report to {out}")


if __name__ == "__main__":
    sys.exit(main())