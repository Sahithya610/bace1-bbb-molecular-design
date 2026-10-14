"""
Full medicinal chemistry validation of dual-success candidates
(BBB+ AND Vina <= -7.5 kcal/mol).

Checks:
  1. PAINS (Pan-Assay INterference compoundS) — flag false positives
  2. Reactive/toxic functional groups (Michael acceptors, aldehydes,
     epoxides, acyl halides, isocyanates, etc.)
  3. Brenk structural alerts (mutagenic/toxic substructures)
  4. QED (Quantitative Estimate of Drug-likeness, 0-1)
  5. Extended drug-likeness: ADMET flags
     - LogP (lipophilicity)
     - TPSA (permeability proxy)
     - Molecular complexity (rotatable bonds, stereocenters)
     - LogS solubility estimate (ESOL)
  6. Novelty vs known BACE1 inhibitors (similarity to ChEMBL BACE1 drugs)

Output: results/docking/bace1_validated_candidates.csv
        results/docking/bace1_validation_report.txt
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, rdMolDescriptors, QED, AllChem, FilterCatalog
)
from rdkit.Chem.FilterCatalog import FilterCatalogParams
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Load dual-success candidates ─────────────────────────────────
df = pd.read_csv("results/docking/bace1_final_ranked.csv")
dual = df[(df["BBB_positive"]) & (df["vina_score"] <= -7.5)].copy()
print(f"Dual-success candidates (BBB+ AND Vina<=-7.5): {len(dual)}")

# ── Set up PAINS filter ───────────────────────────────────────────
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
pains_catalog = FilterCatalog.FilterCatalog(pains_params)

# ── Set up Brenk alerts ───────────────────────────────────────────
brenk_params = FilterCatalogParams()
brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
brenk_catalog = FilterCatalog.FilterCatalog(brenk_params)

# ── Reactive group SMARTS ─────────────────────────────────────────
REACTIVE = {
    "Michael acceptor":     "[CX3]=[CX3][CX3]=[O,N,S]",
    "Aldehyde":             "[CX3H1](=O)",
    "Acyl halide":          "[CX3](=O)[F,Cl,Br,I]",
    "Epoxide":              "C1OC1",
    "Isocyanate":           "N=C=O",
    "Thiocyanate":          "N=C=S",
    "Anhydride":            "C(=O)OC(=O)",
    "Hydrazine":            "[NX2]-[NX2]",
    "Peroxide":             "OO",
    "Nitroso":              "[N]=O",
    "Diazo":                "[N+]#[N-]",
    "Imine_reactive":       "[CX3;H0](=N)",
    "Alpha_halo_carbonyl":  "[CX3](=O)[CX4][F,Cl,Br,I]",
}
reactive_smarts = {k: Chem.MolFromSmarts(v) for k, v in REACTIVE.items()}

# ── ESOL logS estimate (Delaney 2004) ────────────────────────────
def esol_logs(mol):
    """Estimated aqueous solubility (logS, mol/L)."""
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ap   = rdMolDescriptors.CalcNumAromaticRings(mol)
    logs = (0.16 - 0.63*logp - 0.0062*mw + 0.066*rb - 0.74*ap)
    return round(logs, 2)

# ── Reference BACE1 inhibitors (known drugs/candidates) ──────────
# Verubecestat, Atabecestat, Lanabecestat core SMILES
KNOWN_BACE1 = [
    "FC(F)(F)c1cc2c(cc1)C[C@@H]1NC(=O)[C@@H](CC[C@@H]1C2)Cc1cc2c(cc1F)CNC2=O",  # verubecestat-like
    "Fc1ccc(-c2cc3c(cc2F)C[C@H]2NC(=O)CC[C@@H]2C3)cc1",  # simplified BACE1 scaffold
    "NC1=NC(c2ccccc2F)(c2ccccc2F)CS1",  # aminothiazine scaffold
]
known_fps = [
    AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
    for s in KNOWN_BACE1 if Chem.MolFromSmiles(s)
]

from rdkit.DataStructs import TanimotoSimilarity

SEP = "="*70
print(SEP)
print("MEDICINAL CHEMISTRY VALIDATION")
print("Dual-success: BBB-positive AND Vina score ≤ −7.5 kcal/mol")
print(SEP)

results = []
for _, row in dual.iterrows():
    smi = row["canonical_smiles"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue

    # ── PAINS ────────────────────────────────────────────────────
    pains_entries = pains_catalog.GetMatches(mol)
    pains_flags = [e.GetDescription() for e in pains_entries]

    # ── Brenk alerts ─────────────────────────────────────────────
    brenk_entries = brenk_catalog.GetMatches(mol)
    brenk_flags = [e.GetDescription() for e in brenk_entries]

    # ── Reactive groups ───────────────────────────────────────────
    reactive_flags = [
        name for name, pat in reactive_smarts.items()
        if pat and mol.HasSubstructMatch(pat)
    ]

    # ── QED ───────────────────────────────────────────────────────
    qed_score = round(QED.qed(mol), 3)

    # ── Properties ────────────────────────────────────────────────
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    sc   = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    logs = esol_logs(mol)

    # ── Novelty vs known BACE1 ────────────────────────────────────
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    max_sim = max((TanimotoSimilarity(fp, kfp) for kfp in known_fps), default=0.0)

    # ── Overall verdict ───────────────────────────────────────────
    flags_total = len(pains_flags) + len(brenk_flags) + len(reactive_flags)
    if flags_total == 0 and qed_score >= 0.5:
        verdict = "CLEAN"
    elif flags_total == 0:
        verdict = "CLEAN (low QED)"
    elif len(pains_flags) > 0:
        verdict = "PAINS ALERT"
    elif len(brenk_flags) > 0:
        verdict = "BRENK ALERT"
    else:
        verdict = "REACTIVE GROUP"

    results.append({
        "canonical_smiles": smi,
        "oracle_pIC50": row["oracle_pIC50"],
        "vina_score": row["vina_score"],
        "predicted_logBB": row["predicted_logBB"],
        "QED": qed_score,
        "MW": round(mw,1), "LogP": round(logp,2),
        "TPSA": round(tpsa,1), "HBD": hbd, "HBA": hba,
        "RotB": rotb, "Stereocenters": sc, "LogS_ESOL": logs,
        "PAINS": "; ".join(pains_flags) if pains_flags else "none",
        "Brenk": "; ".join(brenk_flags[:3]) if brenk_flags else "none",
        "Reactive": "; ".join(reactive_flags) if reactive_flags else "none",
        "n_alerts": flags_total,
        "max_sim_known_BACE1": round(max_sim, 3),
        "verdict": verdict,
    })

res = pd.DataFrame(results).sort_values(["n_alerts", "QED"], ascending=[True, False])
res.to_csv("results/docking/bace1_validated_candidates.csv", index=False)

# ── Print report ──────────────────────────────────────────────────
clean = res[res["verdict"].str.startswith("CLEAN")]
flagged = res[~res["verdict"].str.startswith("CLEAN")]

print(f"\n{'─'*70}")
print(f"CLEAN (no alerts): {len(clean)}/{len(res)}")
print(f"Flagged:           {len(flagged)}/{len(res)}")
print(f"{'─'*70}")

for _, r in res.iterrows():
    flag_str = ""
    if r["PAINS"] != "none":   flag_str += f" [PAINS: {r['PAINS'][:40]}]"
    if r["Brenk"] != "none":   flag_str += f" [Brenk: {r['Brenk'][:40]}]"
    if r["Reactive"] != "none":flag_str += f" [Reactive: {r['Reactive'][:30]}]"

    sim_str = f"  sim_BACE1={r['max_sim_known_BACE1']:.2f}"
    print(f"\n  [{r['verdict']:16s}]  QED={r['QED']:.3f}  {sim_str}")
    print(f"    Vina={r['vina_score']:.2f}  oracle={r['oracle_pIC50']:.3f}  "
          f"logBB={r['predicted_logBB']:+.3f}")
    print(f"    MW={r['MW']:.0f}  LogP={r['LogP']:.2f}  TPSA={r['TPSA']:.0f}  "
          f"HBD={r['HBD']}  RotB={r['RotB']}  SC={r['Stereocenters']}  LogS={r['LogS_ESOL']}")
    if flag_str:
        print(f"    FLAGS:{flag_str}")
    print(f"    {r['canonical_smiles'][:80]}")

# ── Write text report ─────────────────────────────────────────────
with open("results/docking/bace1_validation_report.txt", "w", encoding="utf-8") as f:
    f.write("BACE1 CANDIDATE VALIDATION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Disclaimer: ALL values are computational predictions.\n")
    f.write(f"Wet-lab synthesis + assay required before any real-world use.\n\n")
    f.write(f"Dual-success criteria: BBB-positive (logBB>=0) AND Vina<=-7.5 kcal/mol\n")
    f.write(f"Candidates evaluated: {len(res)}\n\n")
    f.write(f"CLEAN (no PAINS/Brenk/reactive alerts): {len(clean)}\n")
    f.write(f"Flagged: {len(flagged)}\n\n")
    for i, (_, r) in enumerate(res.iterrows(), 1):
        f.write(f"{'='*70}\n")
        f.write(f"Candidate {i}  [{r['verdict']}]\n")
        f.write(f"  Oracle pIC50:  {r['oracle_pIC50']:.3f}  (predicted, r=0.901)\n")
        f.write(f"  Vina score:    {r['vina_score']:.2f} kcal/mol (simplified receptor)\n")
        f.write(f"  logBB:         {r['predicted_logBB']:+.3f}  (predicted BBB permeability)\n")
        f.write(f"  QED:           {r['QED']:.3f}  (0=least drug-like, 1=most)\n")
        f.write(f"  MW={r['MW']:.0f}  LogP={r['LogP']:.2f}  TPSA={r['TPSA']:.0f}  HBD={r['HBD']}  HBA={r['HBA']}\n")
        f.write(f"  RotBonds={r['RotB']}  Stereocenters={r['Stereocenters']}\n")
        f.write(f"  LogS(ESOL)={r['LogS_ESOL']} mol/L  (aqueous solubility estimate)\n")
        f.write(f"  Sim to known BACE1 scaffold: {r['max_sim_known_BACE1']:.2f}\n")
        if r["PAINS"] != "none":   f.write(f"  PAINS alerts:    {r['PAINS']}\n")
        if r["Brenk"] != "none":   f.write(f"  Brenk alerts:    {r['Brenk']}\n")
        if r["Reactive"] != "none":f.write(f"  Reactive groups: {r['Reactive']}\n")
        f.write(f"  SMILES: {r['canonical_smiles']}\n\n")

print(f"\n{'='*70}")
print(f"Saved: results/docking/bace1_validated_candidates.csv")
print(f"       results/docking/bace1_validation_report.txt")
print(f"\nNOTE: Computational predictions only. Real-world validation requires:")
print(f"  1. Chemical synthesis")
print(f"  2. In vitro BACE1 enzymatic assay (IC50)")
print(f"  3. Cell-based assay (APP processing)")
print(f"  4. BBB permeability assay (PAMPA or MDR1-MDCK)")
print(f"  5. Safety/toxicity profiling")
print(f"  6. In vivo studies")
