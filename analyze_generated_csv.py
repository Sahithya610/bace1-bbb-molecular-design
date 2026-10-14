r"""
analyze_generated_csv.py  (READ-ONLY: Step 3, CSV-level part)

Run from the project root:
    python analyze_generated_csv.py                      # analyzes results\*.csv
    python analyze_generated_csv.py results\a.csv results\b.csv

For every CSV that has a SMILES column it reports parse rate, uniqueness,
dummy-atom ('*') count, fragment count, element presence (O, S, N, F, Cl),
size statistics, and summaries of any pIC50 / logBB / prediction columns.
At the end it prints SMILES overlap between files (file lineage check).

Nothing is modified. The report is written to results\step3_csv_report.txt.
"""
import glob
import os
import sys
import time
from collections import Counter

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski

RDLogger.DisableLog("rdApp.*")
LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def find_smiles_col(df):
    for c in df.columns:
        if c.lower() in ("smiles", "canonical_smiles", "smi"):
            return c
    for c in df.columns:
        if "smiles" in c.lower():
            return c
    return None


def analyze(path):
    log("\n" + "=" * 72)
    log(f"{path}  (modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))})")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log(f"  could not read: {e}")
        return None
    log(f"  rows={len(df)}  columns={list(df.columns)}")
    col = find_smiles_col(df)
    if col is None:
        log("  no SMILES column; skipped")
        return None

    smis = df[col].dropna().astype(str)
    log(f"  SMILES column '{col}': {len(smis)} non-empty values")
    n_star = int(smis.str.contains("*", regex=False).sum())
    n_frag = int(smis.str.contains(".", regex=False).sum())
    log(f"  containing '*' (dummy/padding atom): {n_star}")
    log(f"  multi-fragment (contains '.'):       {n_frag}")

    mols, canon = [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
            canon.append(Chem.MolToSmiles(m))
    n = len(mols)
    log(f"  parse as valid molecules: {n}/{len(smis)}  unique canonical: {len(set(canon))}")
    if n == 0:
        return set()

    elem_atoms, elem_mols = Counter(), Counter()
    heavy, mw, n_n, hbd = [], [], [], []
    for m in mols:
        syms = [a.GetSymbol() for a in m.GetAtoms()]
        elem_atoms.update(syms)
        elem_mols.update(set(syms))
        heavy.append(m.GetNumHeavyAtoms())
        mw.append(Descriptors.MolWt(m))
        n_n.append(syms.count("N"))
        hbd.append(Lipinski.NumHDonors(m))
    log("  molecules containing element (count, %):")
    for e in ("C", "N", "O", "S", "F", "Cl", "Br", "P", "I"):
        log(f"    {e:>2}: {elem_mols.get(e, 0):5d}  {100 * elem_mols.get(e, 0) / n:6.2f}%")
    log(f"  other elements present: {sorted(set(elem_atoms) - set('C N O S F Cl Br P I'.split()))}")
    log(f"  atom totals: {dict(elem_atoms)}")
    s = pd.Series
    log(f"  heavy atoms mean/min/max: {s(heavy).mean():.2f}/{min(heavy)}/{max(heavy)}")
    log(f"  mol weight mean: {s(mw).mean():.2f}   mean N atoms: {s(n_n).mean():.2f}   mean HBD: {s(hbd).mean():.2f}")

    for c in df.columns:
        lc = c.lower()
        if any(t in lc for t in ("pic50", "logbb", "pred", "bace", "bbb")) and pd.api.types.is_numeric_dtype(df[c]):
            v = df[c].dropna()
            if len(v):
                log(f"  {c}: mean={v.mean():.4f} median={v.median():.4f} min={v.min():.4f} max={v.max():.4f} (n={len(v)})")
    return set(canon)


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join("results", "*.csv")))
    sets = {}
    for p in paths:
        r = analyze(p)
        if r:
            sets[p] = r

    names = list(sets)
    if len(names) > 1:
        log("\n" + "=" * 72)
        log("Canonical-SMILES overlap between files (lineage check):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                inter = len(sets[names[i]] & sets[names[j]])
                if inter:
                    log(f"  {inter:5d} shared | {os.path.basename(names[i])} ({len(sets[names[i]])}) vs "
                        f"{os.path.basename(names[j])} ({len(sets[names[j]])})")

    out = os.path.join("results", "step3_csv_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))
    log(f"\nSaved report to {out}")


if __name__ == "__main__":
    main()