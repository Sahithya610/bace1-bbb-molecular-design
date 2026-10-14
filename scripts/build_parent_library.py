"""
Build and validate the Parent Library for Targeted Joint Optimization.
Groups:
- Group A: 6 unique candidates whose oracle-predicted pIC50 >= 7.6576
- Group B: 5 unique candidates with pIC50 >= 7.0 and predicted logBB >= 0.0
- Group C: Promising clean candidates (highlighting Candidate #3 and #5, plus clean #1308)
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, Crippen, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

# Model paths
BACE1_MODEL_PATH = "results/bace1_activity_model/bace1_activity_model.joblib"
BBB_MODEL_PATH = "results/bbb_activity_model/bbb_model.joblib"
BACE1_TRAIN_TABLE = "results/bace1_activity_model/bace1_activity_training_table.csv"
JP_TRAIN_SMI = "druggen_data/bace1_joint_positive_train.smi"

OUT_DIR = Path("results/targeted_joint_optimization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load models
bace1_model = joblib.load(BACE1_MODEL_PATH)
bbb_model = joblib.load(BBB_MODEL_PATH)

# Load training molecules for similarity and duplicate checks
df_bace1_train = pd.read_csv(BACE1_TRAIN_TABLE)
bace1_train_smiles = df_bace1_train["canonical_smiles"].dropna().tolist()

bace1_train_fps = []
for smi in bace1_train_smiles:
    m = Chem.MolFromSmiles(smi)
    if m:
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        bace1_train_fps.append(fp)

# Load FilterCatalogs
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
pains_catalog = FilterCatalog.FilterCatalog(pains_params)

brenk_params = FilterCatalogParams()
brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
brenk_catalog = FilterCatalog.FilterCatalog(brenk_params)

REACTIVE = {
    "Michael acceptor": "[CX3]=[CX3][CX3]=[O,N,S]",
    "Aldehyde": "[CX3H1](=O)",
    "Acyl halide": "[CX3](=O)[F,Cl,Br,I]",
    "Epoxide": "C1OC1",
    "Isocyanate": "N=C=O",
    "Thiocyanate": "N=C=S",
    "Anhydride": "C(=O)OC(=O)",
    "Hydrazine": "[NX2]-[NX2]",
    "Peroxide": "OO",
    "Nitroso": "[N]=O",
    "Diazo": "[N+]#[N-]",
    "Imine_reactive": "[CX3;H0](=N)",
    "Alpha_halo_carbonyl": "[CX3](=O)[CX4][F,Cl,Br,I]",
}
reactive_smarts = {k: Chem.MolFromSmarts(v) for k, v in REACTIVE.items()}

# Raw parent records
RAW_PARENTS = [
    # Group A
    {
        "parent_id": "PARENT_A1",
        "group": "Group A",
        "source_reference": "results/high_activity_candidate_audit.csv rank 1 (50k pilot)",
        "input_smiles": "CC(C)(C)COc1cnc2c(c1)C1(COC(N)=N1)c1cc(-c3cncnc3)ccc1O2",
        "selection_reason": "High predicted pIC50 (7.904), close to logBB boundary (-0.020). Known stereochemical duplicate of training set.",
    },
    {
        "parent_id": "PARENT_A2",
        "group": "Group A",
        "source_reference": "results/high_activity_candidate_audit.csv rank 2 (50k pilot)",
        "input_smiles": "COc1cnc(C(O)Nc2ccc3c(c2)C2(CCSC(N)=N2)c2cc(-c4ccnc(F)c4)nc(F)c2O3)cn1",
        "selection_reason": "High predicted pIC50 (7.736), amino-thiazine warhead, novel structure.",
    },
    {
        "parent_id": "PARENT_A3",
        "group": "Group A / C (Priority Clean)",
        "source_reference": "results/high_activity_candidate_audit.csv rank 3 (50k pilot)",
        "input_smiles": "CCC1CC2(CCC1OC)Cc1ccc(-c3cncnc3)cc1C21COC(N)=N1",
        "selection_reason": "Highest-priority clean candidate: pIC50=7.749, logBB=-0.032 (only 0.032 from 0.0), QED=0.865, no PAINS/Brenk alerts, novel.",
    },
    {
        "parent_id": "PARENT_A4",
        "group": "Group A",
        "source_reference": "results/high_activity_candidate_audit.csv rank 4 (50k pilot)",
        "input_smiles": "CCC(O)NC(Cc1cccc(-c2nccs2)c1)C(O)CNC1CC2(CCC2)Oc2ncc(CC(C)(C)C)cc21",
        "selection_reason": "High predicted pIC50 (7.713), hydroxyethylamine transition-state isostere.",
    },
    {
        "parent_id": "PARENT_A5",
        "group": "Group A / C (Priority Clean)",
        "source_reference": "results/high_activity_candidate_audit.csv rank 5 (50k pilot)",
        "input_smiles": "CCCc1cc(-c2ccc3c(c2)C2(N=C(C)C(N)=N2)C2(CCC(OC)CC2)C3)ccn1",
        "selection_reason": "Second-priority clean candidate: pIC50=7.740, logBB=-0.224, QED=0.777, distinct scaffold from A3, no PAINS/Brenk alerts, novel.",
    },
    {
        "parent_id": "PARENT_A6",
        "group": "Group A",
        "source_reference": "results/high_activity_candidate_audit.csv rank 6 (50k pilot)",
        "input_smiles": "NC1=NC2(CO1)c1cc(-c3cccnc3F)ccc1Oc1cnc(C3CCCOC3)cc12",
        "selection_reason": "High predicted pIC50 (7.921), closest to logBB boundary (-0.014). Stereochemical duplicate of training set.",
    },
    # Group B
    {
        "parent_id": "PARENT_B1",
        "group": "Group B",
        "source_reference": "results/baseline_screen_50000_predictions.csv sample_id 20491",
        "input_smiles": "COC1CCC2(CC1)Cc1ccc(CCC3CC3)cc1C21N=C(N)c2ccccc21",
        "selection_reason": "High activity with BBB positivity: pIC50=7.537 (only 0.12 units from 7.6576), logBB=+0.110. Top Track B candidate.",
    },
    {
        "parent_id": "PARENT_B2",
        "group": "Group B",
        "source_reference": "results/baseline_screen_50000_predictions.csv sample_id 24017",
        "input_smiles": "NC1=NC2(c3ccc(F)cc3F)COC(c3ccc(F)c(F)c3)CC2CS1",
        "selection_reason": "Spiro aminothiazine core: pIC50=7.402, logBB=+0.035, MW=396.4, QED=0.771.",
    },
    {
        "parent_id": "PARENT_B3",
        "group": "Group B",
        "source_reference": "results/baseline_screen_50000_predictions.csv sample_id 35245",
        "input_smiles": "COC1CCC2(CC1)Cc1ccc(-c3cccc(CN)c3)cc1C21N=C(N)c2ccccc21",
        "selection_reason": "Benzamidine spiro core with biaryl cyanomethyl/aminomethyl: pIC50=7.392, logBB=+0.073.",
    },
    {
        "parent_id": "PARENT_B4",
        "group": "Group B",
        "source_reference": "results/baseline_screen_50000_predictions.csv sample_id 38897",
        "input_smiles": "CN1OC2(N=C1N)c1cc(-c3cc(F)cc(CN)c3)ccc1CC21CCc2ccccc2CC1",
        "selection_reason": "Amino-isoxazoline spiro core: pIC50=7.226, logBB=+0.086.",
    },
    {
        "parent_id": "PARENT_B5",
        "group": "Group B",
        "source_reference": "results/baseline_screen_50000_predictions.csv sample_id 48841",
        "input_smiles": "COc1cnc(C(O)Nc2ccc3c(c2)C2(CCSC(N)=N2)c2cc(N4CCOCC4)cc(F)c2O3)cn1",
        "selection_reason": "Morpholine-substituted spiro-thiazine: pIC50=7.129, logBB=+0.028.",
    },
    # Group C additional
    {
        "parent_id": "PARENT_C3",
        "group": "Group C (Clean Baseline)",
        "source_reference": "results/bace1_master_candidate_table.csv sample_id 1308",
        "input_smiles": "CC1CCC(COc2ccc(F)cn2)CN1C(O)c1cc(F)ccc1-c1ncccn1",
        "selection_reason": "Best fully clean baseline candidate: pIC50=6.169, logBB=+0.023, Vina=-8.03, QED=0.643, verified clean.",
    },
]

parent_rows = []
for p in RAW_PARENTS:
    mol = Chem.MolFromSmiles(p["input_smiles"])
    if mol is None:
        raise ValueError(f"Invalid SMILES for {p['parent_id']}")

    canon_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    fp_arr = np.asarray(fp, dtype=np.float32).reshape(1, -1)

    # Predictions
    pred_pIC50 = float(bace1_model.predict(fp_arr)[0])
    pred_logBB = float(bbb_model.predict(fp_arr)[0])

    # Properties
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    qed = Descriptors.qed(mol)

    # Lipinski violations
    ro5_violations = (
        (1 if mw > 500 else 0)
        + (1 if logp > 5.0 else 0)
        + (1 if hbd > 5 else 0)
        + (1 if hba > 10 else 0)
    )

    # Alerts
    pains_matches = [m.GetDescription() for m in pains_catalog.GetMatches(mol)]
    brenk_matches = [m.GetDescription() for m in brenk_catalog.GetMatches(mol)]
    reactive_matches = [k for k, patt in reactive_smarts.items() if mol.HasSubstructMatch(patt)]

    pains_str = "; ".join(pains_matches[:3]) if pains_matches else "none"
    brenk_str = "; ".join(brenk_matches[:3]) if brenk_matches else "none"
    reactive_str = "; ".join(reactive_matches[:3]) if reactive_matches else "none"

    # Nearest training similarity
    sims = DataStructs.BulkTanimotoSimilarity(fp, bace1_train_fps)
    max_tc = max(sims) if sims else 0.0
    nearest_idx = int(np.argmax(sims)) if sims else -1
    nearest_train_smi = bace1_train_smiles[nearest_idx] if nearest_idx >= 0 else "none"

    # Exact duplicate check (ignoring stereochemistry)
    mol_nostereo = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
    smi_nostereo = Chem.MolToSmiles(mol_nostereo, canonical=True)
    is_exact_dup = False
    for t_smi in bace1_train_smiles:
        tm = Chem.MolFromSmiles(t_smi)
        if tm:
            tm_nostereo = Chem.MolFromSmiles(Chem.MolToSmiles(tm, isomericSmiles=False))
            if tm_nostereo and Chem.MolToSmiles(tm_nostereo, canonical=True) == smi_nostereo:
                is_exact_dup = True
                break

    # Stereochemistry status
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    stereo_status = f"{len(chiral_centers)} centers" if chiral_centers else "achiral / unassigned"

    parent_rows.append({
        "parent_id": p["parent_id"],
        "group": p["group"],
        "canonical_isomeric_smiles": canon_smiles,
        "oracle_predicted_pIC50": round(pred_pIC50, 4),
        "bbb_predicted_logBB": round(pred_logBB, 4),
        "passes_pIC50_threshold": pred_pIC50 >= 7.6576,
        "passes_logBB_threshold": pred_logBB >= 0.0,
        "passes_both_jointly": (pred_pIC50 >= 7.6576) and (pred_logBB >= 0.0),
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "HBD": hbd,
        "HBA": hba,
        "RotB": rotb,
        "QED": round(qed, 3),
        "Ro5_violations": ro5_violations,
        "PAINS": pains_str,
        "Brenk": brenk_str,
        "Reactive": reactive_str,
        "max_training_Tc": round(max_tc, 4),
        "exact_training_duplicate": is_exact_dup,
        "nearest_training_smiles": nearest_train_smi,
        "stereochemistry_status": stereo_status,
        "source_reference": p["source_reference"],
        "selection_reason": p["selection_reason"],
    })

df_parents = pd.DataFrame(parent_rows)
out_csv = OUT_DIR / "parent_library.csv"
df_parents.to_csv(out_csv, index=False)
print(f"Parent library saved to {out_csv} ({len(df_parents)} parents)")
print(df_parents[["parent_id", "group", "oracle_predicted_pIC50", "bbb_predicted_logBB", "passes_pIC50_threshold", "passes_logBB_threshold", "QED", "Brenk", "max_training_Tc", "exact_training_duplicate"]].to_string())
