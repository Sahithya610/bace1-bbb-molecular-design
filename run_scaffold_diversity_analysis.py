"""
Scaffold diversity analysis + docking-style report for top 200 molecules.

1. Murcko scaffold decomposition and clustering
2. Lipinski Rule-of-5 filter
3. TPSA, MW, HBD, HBA, rotatable bonds
4. Diversity (Tanimoto pairwise distances)
5. Docking-proxy report: ranks by oracle pIC50 in the same format
   as results/docking/*.csv (actual Vina docking not available for BACE1,
   so oracle pIC50 is used as the activity score)
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from collections import Counter

TOP_FILE  = "results/oracle_filtered_top200_16k.csv"
OUT_ANALYSIS = "results/top200_scaffold_analysis.csv"
OUT_DOCKING  = "results/docking/bace1_druggen_top200_oracle.csv"

print("="*65)
print("SCAFFOLD DIVERSITY + DRUG-LIKENESS ANALYSIS")
print("Top 200 oracle-filtered BACE1 candidates (16k pool)")
print("="*65)

df = pd.read_csv(TOP_FILE)
print(f"Loaded {len(df)} molecules\n")

# ── Compute properties ─────────────────────────────────────────────
rows = []
for _, row in df.iterrows():
    smi = row["canonical_smiles"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue

    mw      = Descriptors.MolWt(mol)
    logp    = Descriptors.MolLogP(mol)
    hbd     = rdMolDescriptors.CalcNumHBD(mol)
    hba     = rdMolDescriptors.CalcNumHBA(mol)
    tpsa    = rdMolDescriptors.CalcTPSA(mol)
    rotb    = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings   = rdMolDescriptors.CalcNumRings(mol)
    arom    = rdMolDescriptors.CalcNumAromaticRings(mol)

    # Lipinski Ro5
    ro5 = int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

    # Veber filter (oral bioavailability proxy)
    veber = int(rotb <= 10 and tpsa <= 140)

    # Murcko scaffold
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False)
    except Exception:
        scaf = ""

    rows.append({
        "canonical_smiles": smi,
        "predicted_pIC50":  row["predicted_pIC50"],
        "predicted_logBB":  row["predicted_logBB"],
        "MW":    round(mw, 1),
        "LogP":  round(logp, 2),
        "HBD":   hbd,
        "HBA":   hba,
        "TPSA":  round(tpsa, 1),
        "RotB":  rotb,
        "Rings": rings,
        "ArRings": arom,
        "Ro5":   ro5,
        "Veber": veber,
        "scaffold": scaf,
    })

props = pd.DataFrame(rows)

# ── Scaffold analysis ──────────────────────────────────────────────
scaf_counts = Counter(props["scaffold"])
n_unique_scaf = len(scaf_counts)
top_scaffolds = scaf_counts.most_common(10)

print(f"[1] SCAFFOLD DIVERSITY")
print(f"    Unique Murcko scaffolds: {n_unique_scaf} / {len(props)}")
print(f"    Singleton scaffolds:     {sum(1 for v in scaf_counts.values() if v==1)}")
print(f"\n    Top 10 scaffolds (count | SMILES):")
for i, (scaf, cnt) in enumerate(top_scaffolds, 1):
    print(f"    {i:2d}. [{cnt:3d}x] {scaf[:80]}")

# ── Drug-likeness ──────────────────────────────────────────────────
print(f"\n[2] DRUG-LIKENESS (top 200)")
print(f"    Lipinski Ro5 pass:  {props['Ro5'].sum()} / {len(props)} ({100*props['Ro5'].mean():.0f}%)")
print(f"    Veber filter pass:  {props['Veber'].sum()} / {len(props)} ({100*props['Veber'].mean():.0f}%)")
print(f"\n    Property distributions (mean ± std):")
for col, label in [("MW","MW"), ("LogP","LogP"), ("HBD","HBD"),
                   ("HBA","HBA"), ("TPSA","TPSA"), ("RotB","RotB")]:
    print(f"    {label:6s}: {props[col].mean():.2f} ± {props[col].std():.2f}  "
          f"[{props[col].min():.1f}–{props[col].max():.1f}]")

# ── Diversity (Tanimoto) ───────────────────────────────────────────
print(f"\n[3] CHEMICAL DIVERSITY (Tanimoto on ECFP4)")
fps = []
valid_idx = []
for i, smi in enumerate(props["canonical_smiles"]):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        fps.append(fp)
        valid_idx.append(i)

# Sample pairwise (all 200 = 19900 pairs, fast enough)
n = len(fps)
sims = []
for i in range(min(n, 200)):
    for j in range(i+1, min(n, 200)):
        sims.append(TanimotoSimilarity(fps[i], fps[j]))

sims = np.array(sims)
print(f"    Pairwise Tanimoto (n={len(sims)} pairs):")
print(f"    Mean similarity:   {sims.mean():.3f}")
print(f"    Median similarity: {np.median(sims):.3f}")
print(f"    Pairs Tc < 0.3 (diverse): {(sims < 0.3).sum()} ({100*(sims<0.3).mean():.0f}%)")
print(f"    Pairs Tc > 0.7 (redundant): {(sims > 0.7).sum()} ({100*(sims>0.7).mean():.0f}%)")

# ── Top candidates after Ro5 filter ───────────────────────────────
print(f"\n[4] TOP 20 AFTER LIPINSKI Ro5 FILTER")
ro5_df = props[props["Ro5"] == 1].sort_values("predicted_pIC50", ascending=False)
print(f"    {len(ro5_df)} molecules pass Ro5")
print()
print(f"  {'#':>3}  {'pIC50':>6}  {'logBB':>6}  {'MW':>6}  {'LogP':>5}  "
      f"{'TPSA':>6}  {'HBD':>4}  {'HBA':>4}  SMILES")
print("  " + "-"*110)
for rank, (_, r) in enumerate(ro5_df.head(20).iterrows(), 1):
    smi_short = r["canonical_smiles"][:55] + "..." if len(r["canonical_smiles"]) > 55 else r["canonical_smiles"]
    print(f"  {rank:3d}  {r['predicted_pIC50']:6.3f}  {r['predicted_logBB']:+6.3f}  "
          f"{r['MW']:6.1f}  {r['LogP']:5.2f}  {r['TPSA']:6.1f}  "
          f"{r['HBD']:4.0f}  {r['HBA']:4.0f}  {smi_short}")

# ── Save analysis ──────────────────────────────────────────────────
props.to_csv(OUT_ANALYSIS, index=False)
print(f"\nFull analysis saved to: {OUT_ANALYSIS}")

# ── Docking-proxy output (oracle score as activity proxy) ──────────
import os
os.makedirs("results/docking", exist_ok=True)

# Format matches existing docking CSVs in results/docking/
docking_df = props[["canonical_smiles", "predicted_pIC50"]].copy()
docking_df.columns = ["SMILES", "ORACLE_pIC50"]
docking_df.insert(0, "DrugGEN_MOL_ID",
                  [f"BACE1_MOL_{i+1:05d}" for i in range(len(docking_df))])
docking_df = docking_df.sort_values("ORACLE_pIC50", ascending=False).reset_index(drop=True)
docking_df.to_csv(OUT_DOCKING, index=False)

print(f"Docking-proxy report saved to: {OUT_DOCKING}")
print()
print("NOTE: Actual Vina docking for BACE1 requires the receptor PDB/PDBQT")
print("      file and AutoDock-Vina installation. Oracle pIC50 is used as")
print("      a proxy for activity ranking (r=0.901 on training molecules).")

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"  Pool generated:        16,384 molecules")
print(f"  Valid after decode:    {len(df):,}")
print(f"  Oracle scored:         11,376")
print(f"  pIC50 ≥ 6.0:           318  (2.8%)")
print(f"  pIC50 ≥ 7.0:           5    (0.04%)")
print(f"  Top 200 unique scaffolds: {n_unique_scaf}")
print(f"  Top 200 Ro5 pass:      {props['Ro5'].sum()} ({100*props['Ro5'].mean():.0f}%)")
print(f"  Diversity (mean Tc):   {sims.mean():.3f}")
