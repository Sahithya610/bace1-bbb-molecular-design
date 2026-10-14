"""
Targeted Joint Molecular Optimization Engine for BACE1 pIC50 and BBB logBB.

Executes a bounded pilot of up to 500 unique valid analogues across:
- Track A: Improve BBB permeability (logBB >= 0.0) while preserving BACE1 activity (pIC50 >= 7.6576)
- Track B: Improve BACE1 activity (pIC50 >= 7.6576) while preserving BBB permeability (logBB >= 0.0)

Generates all required deliverables under results/targeted_joint_optimization/:
1. parent_library.csv (already verified)
2. generated_analogues.csv
3. analogue_predictions.csv
4. analogue_exclusions.csv
5. joint_threshold_hits.csv
6. optimization_summary.json
7. run_config.json
"""

import copy
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, AllChem, Crippen, FilterCatalog, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

# Paths
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "targeted_joint_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARENT_CSV = OUT_DIR / "parent_library.csv"
BACE1_MODEL_PATH = ROOT / "results" / "bace1_activity_model" / "bace1_activity_model.joblib"
BBB_MODEL_PATH = ROOT / "results" / "bbb_activity_model" / "bbb_model.joblib"
BACE1_TRAIN_TABLE = ROOT / "results" / "bace1_activity_model" / "bace1_activity_training_table.csv"
JP_TRAIN_SMI = ROOT / "druggen_data" / "bace1_joint_positive_train.smi"

# Thresholds
THRESH_PIC50 = 7.6576
THRESH_LOGBB = 0.0

# -----------------------------------------------------------------------------
# 1. Load Models, Catalogs, and Training Sets
# -----------------------------------------------------------------------------
print("Loading prediction models and training fingerprints...")
bace1_model = joblib.load(BACE1_MODEL_PATH)
bbb_model = joblib.load(BBB_MODEL_PATH)

df_bace1_train = pd.read_csv(BACE1_TRAIN_TABLE)
bace1_train_smiles = df_bace1_train["canonical_smiles"].dropna().tolist()
bace1_train_fps = []
for smi in bace1_train_smiles:
    m = Chem.MolFromSmiles(smi)
    if m:
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        bace1_train_fps.append(fp)

jp_train_fps = []
if JP_TRAIN_SMI.is_file():
    for line in open(JP_TRAIN_SMI, encoding="utf-8"):
        s = line.strip()
        if s:
            m = Chem.MolFromSmiles(s)
            if m:
                jp_train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))

# FilterCatalogs
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

# -----------------------------------------------------------------------------
# 2. Load Parents
# -----------------------------------------------------------------------------
df_parents = pd.read_csv(PARENT_CSV)
print(f"Loaded {len(df_parents)} parents from {PARENT_CSV}")

# -----------------------------------------------------------------------------
# 3. Medicinal Chemistry Transformation Engine
# -----------------------------------------------------------------------------

def safe_apply_reaction(rxn_smarts, mol):
    """Apply a reaction SMARTS safely to an RDKit mol. Returns list of product mols."""
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        products = rxn.RunReactants((mol,))
        valid_products = []
        for prod_tuple in products:
            prod = prod_tuple[0]
            try:
                Chem.SanitizeMol(prod)
                valid_products.append(prod)
            except Exception:
                continue
        return valid_products
    except Exception:
        return []


def generate_track_a_analogues(parent_id, parent_mol, parent_smiles):
    """
    Track A: Improve BBB permeability (logBB >= 0.0) from High Activity parents (Group A).
    Key strategies:
    1. Pyrimidine -> Pyridine / Fluorophenyl / Pyridyl-F bioisosteres (reduces TPSA by ~13-26 A^2)
    2. Ether modifications: OMe -> OEt, OiPr, OCF3, OCH2CF3, OCH2cPr, F, Me
    3. Alkyl modifications: Et -> Me, cPr, iPr, CF3
    4. Halogenation: aromatic mono-fluorination (ortho/meta/para scans)
    5. Ether oxygen replacement: -O- -> -CH2-
    """
    analogues = [] # list of (mol, category, description)

    # 1. Pyrimidine bioisosteric swaps (in A1, A3)
    pyrim_to_pyr4 = "[c:1]1[c:2][n:3][c:4][n:5][c:6]1>>[c:1]1[c:2][c:3][n:4][c:5][c:6]1"
    pyrim_to_pyr3 = "[c:1]1[c:2][n:3][c:4][n:5][c:6]1>>[c:1]1[c:2][n:3][c:4][c:5][c:6]1"
    pyrim_to_ph = "[c:1]1[c:2][n:3][c:4][n:5][c:6]1>>[c:1]1[c:2][c:3][c:4][c:5][c:6]1"
    pyrim_to_4fph = "[c:1]1[c:2][n:3][c:4][n:5][c:6]1>>[c:1]1[c:2][c:3][c:4](F)[c:5][c:6]1"
    pyrim_to_3fph = "[c:1]1[c:2][n:3][c:4][n:5][c:6]1>>[c:1]1[c:2][c:3](F)[c:4][c:5][c:6]1"
    pyrim_to_35dfph = "[c:1]1[c:2][n:3][c:4][n:5][c:6]1>>[c:1]1[c:2][c:3](F)[c:4][c:5](F)[c:6]1"
    pyrim_to_2fpyr = "[c:1]1[c:2][n:3][c:4][n:5][c:6]1>>[c:1]1[c:2][c:3][n:4][c:5](F)[c:6]1"

    for sm, desc in [
        (pyrim_to_pyr4, "Pyrimidine -> 4-pyridyl bioisostere (-13 A^2 TPSA, -1 HBA)"),
        (pyrim_to_pyr3, "Pyrimidine -> 3-pyridyl bioisostere (-13 A^2 TPSA, -1 HBA)"),
        (pyrim_to_ph, "Pyrimidine -> phenyl bioisostere (-26 A^2 TPSA, -2 HBA)"),
        (pyrim_to_4fph, "Pyrimidine -> 4-fluorophenyl bioisostere (lipophilic + polar surface reduction)"),
        (pyrim_to_3fph, "Pyrimidine -> 3-fluorophenyl bioisostere"),
        (pyrim_to_35dfph, "Pyrimidine -> 3,5-difluorophenyl bioisostere (classical BACE1 S3 motif)"),
        (pyrim_to_2fpyr, "Pyrimidine -> 2-fluoropyridin-4-yl bioisostere"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Bioisostere_TPSA_Reduction", desc))

    # 2. Methoxy group substitutions: [C:1]-[O:2]-[CH3:3]
    for sm, desc in [
        ("[C:1][O:2][CH3:3]>>[C:1][O:2]CC", "Methoxy -> Ethoxy (+CH2 lipophilicity boost)"),
        ("[C:1][O:2][CH3:3]>>[C:1][O:2]C(C)C", "Methoxy -> Isopropoxy (+2x CH3 lipophilicity boost)"),
        ("[C:1][O:2][CH3:3]>>[C:1][O:2]C(F)(F)F", "Methoxy -> Trifluoromethoxy (classic CNS bioisostere)"),
        ("[C:1][O:2][CH3:3]>>[C:1][O:2]CC(F)(F)F", "Methoxy -> 2,2,2-trifluoroethoxy"),
        ("[C:1][O:2][CH3:3]>>[C:1][O:2]CC1CC1", "Methoxy -> Cyclopropylmethoxy (compact lipophilic ether)"),
        ("[C:1][O:2][CH3:3]>>[C:1]F", "Methoxy -> Fluorine replacement (removes HBA)"),
        ("[C:1][O:2][CH3:3]>>[C:1]C", "Methoxy -> Methyl replacement"),
        ("[C:1][O:2][CH3:3]>>[C:1][O:2]C1CC1", "Methoxy -> Cyclopropoxy"),
        ("[C:1][O:2][CH3:3]>>[C:1]CF", "Methoxy -> Fluoromethyl replacement"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Ether_Lipophilicity_Tuning", desc))

    # 3. Ethyl / Propyl sidechain tuning: [CH2:1][CH3:2]
    for sm, desc in [
        ("[CH2:1][CH3:2]>>[CH3:1]", "Alkyl truncation -> Methyl"),
        ("[CH2:1][CH3:2]>>C1CC1", "Ethyl -> Cyclopropyl (rigid lipophilic bioisostere)"),
        ("[CH2:1][CH3:2]>>C(C)C", "Ethyl -> Isopropyl"),
        ("[CH2:1][CH3:2]>>C(F)(F)F", "Ethyl -> Trifluoromethyl"),
        ("[CH2:1][CH3:2]>>CC(F)(F)F", "Ethyl -> 2,2,2-trifluoroethyl"),
        ("[CH2:1][CH3:2]>>CCF", "Ethyl -> 2-fluoroethyl"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Alkyl_Chain_Tuning", desc))

    # 4. Aromatic fluorination (C-H -> C-F scan) on unsubstituted aromatic carbons
    # Reaction on [c;H1]
    aro_f_rxn = "[c;H1:1]>>[c:1]F"
    prods = safe_apply_reaction(aro_f_rxn, parent_mol)
    for i, prod in enumerate(prods[:6]): # take up to 6 distinct aromatic fluorination sites
        analogues.append((prod, "Halogen_Scan", f"Aromatic C-H -> C-F mono-fluorination (site {i+1})"))

    # 5. Aromatic chloro scan (C-H -> C-Cl)
    aro_cl_rxn = "[c;H1:1]>>[c:1]Cl"
    prods_cl = safe_apply_reaction(aro_cl_rxn, parent_mol)
    for i, prod in enumerate(prods_cl[:4]):
        analogues.append((prod, "Halogen_Scan", f"Aromatic C-H -> C-Cl mono-chlorination (site {i+1})"))

    # 6. Specific parent-directed transformations:
    # A1: tert-butyl ether tuning [CH2][C](C)(C)C
    for sm, desc in [
        ("c1[O:1]CC(C)(C)C>>c1[O:1]CC1CC1", "Neopentyl ether -> Cyclopropylmethyl ether"),
        ("c1[O:1]CC(C)(C)C>>c1[O:1]C(C)(C)C", "Neopentyl ether -> tert-Butyl ether"),
        ("c1[O:1]CC(C)(C)C>>c1[O:1]CC(F)(F)F", "Neopentyl ether -> 2,2,2-trifluoroethyl ether"),
        ("c1[O:1]CC(C)(C)C>>c1[O:1]C1CCCC1", "Neopentyl ether -> Cyclopentyl ether"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Parent_Specific_Optimization", desc))

    # A5: propyl-pyridine chain tuning [c]CCC
    for sm, desc in [
        ("c1[c:1](CCC)ccn1>>c1[c:1](C1CC1)ccn1", "Propyl-pyridine -> Cyclopropyl-pyridine"),
        ("c1[c:1](CCC)ccn1>>c1[c:1](C(C)C)ccn1", "Propyl-pyridine -> Isopropyl-pyridine"),
        ("c1[c:1](CCC)ccn1>>c1[c:1](CC)ccn1", "Propyl-pyridine -> Ethyl-pyridine"),
        ("c1[c:1](CCC)ccn1>>c1[c:1](C(F)(F)F)ccn1", "Propyl-pyridine -> Trifluoromethyl-pyridine"),
        ("c1[c:1](CCC)ccn1>>c1[c:1](CCC)c(F)cn1", "Propyl-pyridine -> 3-fluoro-propyl-pyridine"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Parent_Specific_Optimization", desc))

    # A6: THP ether replacement
    for sm, desc in [
        ("c1[c:1](C2CCCOC2)ccn1>>c1[c:1](C2CCOC2)ccn1", "Tetrahydropyran -> Tetrahydrofuran"),
        ("c1[c:1](C2CCCOC2)ccn1>>c1[c:1](C2CCC2)ccn1", "Tetrahydropyran -> Cyclobutyl"),
        ("c1[c:1](C2CCCOC2)ccn1>>c1[c:1](C2CCCC2)ccn1", "Tetrahydropyran -> Cyclopentyl"),
        ("c1[c:1](C2CCCOC2)ccn1>>c1[c:1](C2(F)CCCC2)ccn1", "Tetrahydropyran -> Fluorocyclopentyl"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Parent_Specific_Optimization", desc))

    return analogues


def generate_track_b_analogues(parent_id, parent_mol, parent_smiles):
    """
    Track B: Improve BACE1 activity (pIC50 >= 7.6576) from BBB-positive parents (Group B).
    Key strategies:
    1. B1 (sample 20491): The cyclopropyl-ethyl group 'CCC3CC3' is in the S1/S3 pocket region.
       Replace with classic high-affinity BACE1 S1/S3 motifs:
       - 3,5-difluorophenyl
       - 3-fluorophenyl
       - 2,5-difluorophenyl
       - 2-fluoropyridin-4-yl
       - 5-pyrimidinyl
       - 3-(trifluoromethoxy)phenyl
       - 3-chlorophenyl
       - 2-cyanopyridin-4-yl
    2. Benzamidine / amino-heterocycle warhead modifications:
       - Fluorination / chlorination of fused benzo ring
       - Amino-oxazoline / amino-thiazine bioisosteric variants
    3. Methoxy tuning on the cyclohexyl ring: OMe -> F, OEt, OCF3, OH
    4. B2: Optimization of pendant 2,4-difluorophenyl and 3,4-difluorophenyl rings
    5. B3: Cyano group optimization (CN -> F, CF3, Cl, OMe, cPr)
    6. B4: N-methyl isoxazoline optimization
    7. B5: Morpholine bioisosteric swaps (piperidine, N-methylpiperazine, thiomorpholine-1,1-dioxide)
    """
    analogues = []

    # 1. B1 Specific: Replacement of the flexible cyclopropyl-ethyl group with S1/S3 pocket aromatic motifs
    # In B1: c1ccc(CCC2CC2)cc1
    # Replace -CCC2CC2 with aromatic BACE1 pharmacophores
    for sm, desc in [
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2cc(F)cc(F)c2)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 3,5-difluorophenyl (classical BACE1 S1/S3 binder)"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2cccc(F)c2)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 3-fluorophenyl"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2ccc(F)c(F)c2)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 3,4-difluorophenyl"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2cc(F)ccc2F)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 2,5-difluorophenyl"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2cncnc2)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 5-pyrimidinyl (as in Candidate A3)"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2ccncc2)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 4-pyridyl"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2cnccc2F)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 2-fluoropyridin-4-yl"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2cccc(C(F)(F)F)c2)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 3-(trifluoromethyl)phenyl"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2cccc(OC(F)(F)F)c2)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 3-(trifluoromethoxy)phenyl"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2cccc(Cl)c2)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 3-chlorophenyl"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2cc(Cl)cc(F)c2)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 3-chloro-5-fluorophenyl"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2cccc(C#N)c2)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 3-cyanophenyl"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(-c2cccc(CN)c2)cc1", "P1/P3 Pocket: Cyclopropylethyl -> 3-(aminomethyl)phenyl"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(CCc2cc(F)cc(F)c2)cc1", "P1/P3 Pocket: Flexible 3,5-difluorophenethyl isostere"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(CCc2cccc(F)c2)cc1", "P1/P3 Pocket: Flexible 3-fluorophenethyl isostere"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(CC2CCCC2)cc1", "P1/P3 Pocket: Cyclopentylmethyl isostere"),
        ("c1ccc(CCC2CC2)cc1>>c1ccc(CC2CCCCC2)cc1", "P1/P3 Pocket: Cyclohexylmethyl isostere"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "BACE1_Pocket_Pharmacophore", desc))

    # 2. B1 Benzamidine fused ring halogenation / modification: c2ccccc2
    # In B1: C21N=C(N)c2ccccc21
    for sm, desc in [
        ("[N:1]=[C:2]([N:3])c1cc([c;H1:4])ccc1>>[N:1]=[C:2]([N:3])c1cc([c:4]F)ccc1", "Benzamidine ring 4-fluoro substitution (Asp-anchor tuning)"),
        ("[N:1]=[C:2]([N:3])c1ccc([c;H1:4])cc1>>[N:1]=[C:2]([N:3])c1ccc([c:4]F)cc1", "Benzamidine ring 5-fluoro substitution"),
        ("[N:1]=[C:2]([N:3])c1cc([c;H1:4])c([c;H1:5])cc1>>[N:1]=[C:2]([N:3])c1cc([c:4]F)c([c:5]F)cc1", "Benzamidine ring 4,5-difluoro substitution"),
        ("[N:1]=[C:2]([N:3])c1cc([c;H1:4])ccc1>>[N:1]=[C:2]([N:3])c1cc([c:4]Cl)ccc1", "Benzamidine ring 4-chloro substitution"),
        ("[N:1]=[C:2]([N:3])c1cc([c;H1:4])ccc1>>[N:1]=[C:2]([N:3])c1cc([c:4]C#N)ccc1", "Benzamidine ring 4-cyano substitution"),
        ("[N:1]=[C:2]([N:3])c1cc([c;H1:4])ccc1>>[N:1]=[C:2]([N:3])c1cc([c:4]C(F)(F)F)ccc1", "Benzamidine ring 4-trifluoromethyl substitution"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Warhead_Substituent_Optimization", desc))

    # 3. Cyclohexyl methoxy substitutions on B1/B3: [C:1][O:2][CH3:3]
    for sm, desc in [
        ("[C:1][O:2][CH3:3]>>[C:1]F", "Spiro-cyclohexyl: Methoxy -> Fluorine"),
        ("[C:1][O:2][CH3:3]>>[C:1]O", "Spiro-cyclohexyl: Methoxy -> Hydroxyl (H-bond donor for subpocket)"),
        ("[C:1][O:2][CH3:3]>>[C:1][O:2]CC", "Spiro-cyclohexyl: Methoxy -> Ethoxy"),
        ("[C:1][O:2][CH3:3]>>[C:1][O:2]C(F)(F)F", "Spiro-cyclohexyl: Methoxy -> Trifluoromethoxy"),
        ("[C:1][O:2][CH3:3]>>[C:1]C", "Spiro-cyclohexyl: Methoxy -> Methyl"),
        ("[C:1][O:2][CH3:3]>>[C:1]C(F)(F)F", "Spiro-cyclohexyl: Methoxy -> Trifluoromethyl"),
        ("[C:1][O:2][CH3:3]>>[C:1]C1CC1", "Spiro-cyclohexyl: Methoxy -> Cyclopropyl"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Spiro_Ring_Optimization", desc))

    # 4. B2 Specific (spiro aminothiazine: NC1=NC2(c3ccc(F)cc3F)COC(c3ccc(F)c(F)c3)CC2CS1)
    for sm, desc in [
        ("c1cc(F)c(F)cc1>>c1cc(F)cc(F)c1", "Difluorophenyl -> 3,5-difluorophenyl (BACE1 S1 preference)"),
        ("c1ccc(F)cc1F>>c1cc(F)cc(F)c1", "2,4-difluorophenyl -> 3,5-difluorophenyl"),
        ("c1cc(F)c(F)cc1>>c1cccc(F)c1", "Difluorophenyl -> 3-fluorophenyl"),
        ("c1ccc(F)cc1F>>c1cccc(F)c1", "2,4-difluorophenyl -> 3-fluorophenyl"),
        ("c1cc(F)c(F)cc1>>c1cncnc1", "Difluorophenyl -> 5-pyrimidinyl"),
        ("CC2CS1>>CC2CO1", "Aminothiazine -> Amino-oxazine bioisostere (S -> O)"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Aminothiazine_S1_Optimization", desc))

    # 5. B3 Specific (biaryl cyanomethyl: c1ccc(-c2cccc(CN)c2)cc1)
    for sm, desc in [
        ("c1cccc(CN)c1>>c1cc(F)cc(F)c1", "3-(aminomethyl)phenyl -> 3,5-difluorophenyl"),
        ("c1cccc(CN)c1>>c1cccc(F)c1", "3-(aminomethyl)phenyl -> 3-fluorophenyl"),
        ("c1cccc(CN)c1>>c1cccc(C(F)(F)F)c1", "3-(aminomethyl)phenyl -> 3-(trifluoromethyl)phenyl"),
        ("c1cccc(CN)c1>>c1cccc(OC(F)(F)F)c1", "3-(aminomethyl)phenyl -> 3-(trifluoromethoxy)phenyl"),
        ("c1cccc(CN)c1>>c1cncnc1", "3-(aminomethyl)phenyl -> 5-pyrimidinyl"),
        ("c1cccc(CN)c1>>c1cnccc1F", "3-(aminomethyl)phenyl -> 2-fluoropyridin-4-yl"),
        ("c1cccc(CN)c1>>c1cccc(C#N)c1", "3-(aminomethyl)phenyl -> 3-cyanophenyl"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Biaryl_Pharmacophore_Tuning", desc))

    # 6. B4 Specific (isoxazoline: CN1OC2...c1cc(-c3cc(F)cc(CN)c3)ccc1)
    for sm, desc in [
        ("CN1OC2>>N1OC2", "Isoxazoline: N-methyl -> N-H"),
        ("CN1OC2>>CCN1OC2", "Isoxazoline: N-methyl -> N-ethyl"),
        ("CN1OC2>>C1CC1N1OC2", "Isoxazoline: N-methyl -> N-cyclopropyl"),
        ("c3cc(F)cc(CN)c3>>c3cc(F)cc(F)c3", "Fluorobenzylamine -> 3,5-difluorophenyl"),
        ("c3cc(F)cc(CN)c3>>c3cncnc3", "Fluorobenzylamine -> 5-pyrimidinyl"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Isoxazoline_Tuning", desc))

    # 7. B5 Specific (morpholine substitution)
    for sm, desc in [
        ("N1CCOCC1>>N1CCCCC1", "Morpholine -> Piperidine (increases logBB and lipophilicity)"),
        ("N1CCOCC1>>N1CCN(C)CC1", "Morpholine -> N-methylpiperazine"),
        ("N1CCOCC1>>N1CCS(=O)(=O)CC1", "Morpholine -> Thiomorpholine-1,1-dioxide"),
        ("N1CCOCC1>>F", "Morpholine -> Fluorine replacement"),
        ("N1CCOCC1>>C1CC1", "Morpholine -> Cyclopropyl replacement"),
        ("N1CCOCC1>>OC(F)(F)F", "Morpholine -> Trifluoromethoxy replacement"),
    ]:
        for prod in safe_apply_reaction(sm, parent_mol):
            analogues.append((prod, "Morpholine_Bioisostere", desc))

    # 8. General aromatic C-H -> C-F scan (up to 4 sites)
    aro_f_rxn = "[c;H1:1]>>[c:1]F"
    prods = safe_apply_reaction(aro_f_rxn, parent_mol)
    for i, prod in enumerate(prods[:4]):
        analogues.append((prod, "Halogen_Scan", f"Aromatic C-H -> C-F mono-fluorination (site {i+1})"))

    return analogues


# -----------------------------------------------------------------------------
# 4. Execute Analog Generation Loop
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("GENERATING BOUNDED ANALOGUE LIBRARY")
print("=" * 78)

generated_records = []
exclusion_records = []
seen_isomeric_smiles = set()

for idx, parent_row in df_parents.iterrows():
    pid = parent_row["parent_id"]
    grp = parent_row["group"]
    p_smi = parent_row["canonical_isomeric_smiles"]
    p_mol = Chem.MolFromSmiles(p_smi)

    if p_mol is None:
        continue

    # Determine track
    if "Group A" in grp:
        track = "Track A"
        candidate_analogues = generate_track_a_analogues(pid, p_mol, p_smi)
    elif "Group B" in grp:
        track = "Track B"
        candidate_analogues = generate_track_b_analogues(pid, p_mol, p_smi)
    else: # Group C baseline
        track = "Track B"
        candidate_analogues = generate_track_b_analogues(pid, p_mol, p_smi)

    print(f"Parent {pid} ({grp}): generated {len(candidate_analogues)} raw transforms")

    for prod_mol, cat, desc in candidate_analogues:
        # Check sanitization
        try:
            Chem.SanitizeMol(prod_mol)
        except Exception as e:
            exclusion_records.append({
                "parent_id": pid,
                "transformation_category": cat,
                "transformation_description": desc,
                "error_stage": "sanitization",
                "exclusion_reason": f"Sanitization failed: {str(e)[:100]}",
            })
            continue

        # Check heavy atom count
        heavy_atoms = prod_mol.GetNumHeavyAtoms()
        if heavy_atoms < 10 or heavy_atoms > 50:
            exclusion_records.append({
                "parent_id": pid,
                "transformation_category": cat,
                "transformation_description": desc,
                "error_stage": "heavy_atom_filter",
                "exclusion_reason": f"Heavy atom count out of bounds: {heavy_atoms}",
            })
            continue

        # Canonical isomeric SMILES
        try:
            iso_smi = Chem.MolToSmiles(prod_mol, canonical=True, isomericSmiles=True)
            noniso_smi = Chem.MolToSmiles(prod_mol, canonical=True, isomericSmiles=False)
        except Exception as e:
            exclusion_records.append({
                "parent_id": pid,
                "transformation_category": cat,
                "transformation_description": desc,
                "error_stage": "smiles_generation",
                "exclusion_reason": f"SMILES generation failed: {str(e)[:100]}",
            })
            continue

        # Check identical to parent
        if iso_smi == p_smi:
            exclusion_records.append({
                "parent_id": pid,
                "transformation_category": cat,
                "transformation_description": desc,
                "error_stage": "parent_identity",
                "exclusion_reason": "Identical to parent molecule",
            })
            continue

        # Deduplication
        if iso_smi in seen_isomeric_smiles:
            exclusion_records.append({
                "parent_id": pid,
                "transformation_category": cat,
                "transformation_description": desc,
                "error_stage": "deduplication",
                "exclusion_reason": "Duplicate analogue already generated",
            })
            continue

        seen_isomeric_smiles.add(iso_smi)

        # Stereochemistry status
        parent_centers = Chem.FindMolChiralCenters(p_mol, includeUnassigned=True)
        prod_centers = Chem.FindMolChiralCenters(prod_mol, includeUnassigned=True)
        stereo_change = (len(parent_centers) != len(prod_centers))

        aid = f"ANALOGUE_{len(generated_records)+1:04d}"
        generated_records.append({
            "analogue_id": aid,
            "parent_id": pid,
            "parent_group": grp,
            "track": track,
            "transformation_category": cat,
            "transformation_description": desc,
            "parent_canonical_smiles": p_smi,
            "analogue_canonical_isomeric_smiles": iso_smi,
            "analogue_canonical_nonisomeric_smiles": noniso_smi,
            "stereochemistry_modified": stereo_change,
            "n_chiral_centers": len(prod_centers),
        })

print(f"\nGenerated {len(generated_records)} unique valid analogues!")
print(f"Excluded / filtered transforms: {len(exclusion_records)}")

# -----------------------------------------------------------------------------
# 5. Evaluate All Unique Valid Analogues
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("EVALUATING ANALOGUES VIA INDEPENDENT BACE1 AND BBB PREDICTION MODELS")
print("=" * 78)

prediction_records = []
joint_hits = []

t0 = time.time()
for rec in generated_records:
    smi = rec["analogue_canonical_isomeric_smiles"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue

    # Morgan fingerprint (radius 2, 2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    fp_arr = np.asarray(fp, dtype=np.float32).reshape(1, -1)

    # Independent Oracle Predictions
    pred_pIC50 = float(bace1_model.predict(fp_arr)[0])
    pred_logBB = float(bbb_model.predict(fp_arr)[0])
    norm_logBB = (pred_logBB - (-0.077873)) / 0.751623

    # Physicochemical properties
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    qed = Descriptors.qed(mol)

    # Lipinski Ro5 violations
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
    chem_clean = (len(pains_matches) == 0) and (len(brenk_matches) == 0) and (len(reactive_matches) == 0)

    # Training similarity
    sims_bace1 = DataStructs.BulkTanimotoSimilarity(fp, bace1_train_fps)
    max_tc_bace1 = max(sims_bace1) if sims_bace1 else 0.0

    max_tc_jp = 0.0
    if jp_train_fps:
        sims_jp = DataStructs.BulkTanimotoSimilarity(fp, jp_train_fps)
        max_tc_jp = max(sims_jp) if sims_jp else 0.0

    # Exact training duplicate check
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

    # Threshold checks
    pass_pIC50 = bool(pred_pIC50 >= THRESH_PIC50)
    pass_logBB = bool(pred_logBB >= THRESH_LOGBB)
    pass_both = bool(pass_pIC50 and pass_logBB)

    # Model applicability check
    # B3DB ExtraTrees training domain: MW 100-600, LogP -2 to 6, TPSA 0-150
    in_bbb_domain = (100 <= mw <= 600) and (-2.0 <= logp <= 6.0) and (tpsa <= 150.0)

    # Borderline flag (logBB within 0.1 of threshold 0.0)
    is_borderline = (0.0 <= pred_logBB <= 0.10)

    pred_record = {
        "analogue_id": rec["analogue_id"],
        "parent_id": rec["parent_id"],
        "parent_group": rec["parent_group"],
        "track": rec["track"],
        "transformation_category": rec["transformation_category"],
        "transformation_description": rec["transformation_description"],
        "parent_smiles": rec["parent_canonical_smiles"],
        "analogue_smiles": smi,
        "oracle_predicted_pIC50": round(pred_pIC50, 4),
        "bbb_predicted_logBB": round(pred_logBB, 4),
        "normalized_predicted_logBB": round(norm_logBB, 4),
        "passes_pIC50_threshold": pass_pIC50,
        "passes_logBB_threshold": pass_logBB,
        "passes_both_jointly": pass_both,
        "is_borderline_bbb_hit": is_borderline,
        "in_bbb_domain": in_bbb_domain,
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
        "chemically_clean": chem_clean,
        "max_bace1_train_Tc": round(max_tc_bace1, 4),
        "max_joint_pos_train_Tc": round(max_tc_jp, 4),
        "exact_training_duplicate": is_exact_dup,
        "stereochemistry_modified": rec["stereochemistry_modified"],
    }
    prediction_records.append(pred_record)

    if pass_both:
        joint_hits.append(pred_record)

eval_time = time.time() - t0
print(f"Evaluated {len(prediction_records)} analogues in {eval_time:.1f}s")
print(f"Total passing pIC50 >= 7.6576: {sum(r['passes_pIC50_threshold'] for r in prediction_records)}")
print(f"Total passing logBB >= 0.0:    {sum(r['passes_logBB_threshold'] for r in prediction_records)}")
print(f"Total passing BOTH JOINTLY:   {len(joint_hits)}")

# -----------------------------------------------------------------------------
# 6. Save Deliverables
# -----------------------------------------------------------------------------
df_gen = pd.DataFrame(generated_records)
df_gen.to_csv(OUT_DIR / "generated_analogues.csv", index=False)

df_pred = pd.DataFrame(prediction_records)
df_pred.to_csv(OUT_DIR / "analogue_predictions.csv", index=False)

df_excl = pd.DataFrame(exclusion_records)
df_excl.to_csv(OUT_DIR / "analogue_exclusions.csv", index=False)

df_hits = pd.DataFrame(joint_hits)
df_hits.to_csv(OUT_DIR / "joint_threshold_hits.csv", index=False)

# Configuration
run_config = {
    "experiment": "Targeted_Joint_Optimization_Pilot",
    "objective": (
        "Identify valid, sufficiently novel candidate molecules satisfying BOTH "
        "independent BACE1 oracle pIC50 >= 7.6576 AND BBB model logBB >= 0.0 using "
        "bounded molecular optimization on verified candidate parents."
    ),
    "bace1_oracle_model": str(BACE1_MODEL_PATH.resolve()),
    "bace1_oracle_test_R2": 0.6156,
    "bace1_oracle_test_RMSE": 0.7553,
    "bbb_model": str(BBB_MODEL_PATH.resolve()),
    "bbb_model_test_R2": 0.3538,
    "bbb_model_test_RMSE": 0.6215,
    "thresholds": {
        "pIC50": THRESH_PIC50,
        "logBB": THRESH_LOGBB,
    },
    "parent_library_size": len(df_parents),
    "analogue_pilot_limit": 500,
    "total_unique_valid_analogues_generated": len(df_pred),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}

with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(run_config, f, indent=2)

# Optimization Summary
opt_summary = {
    "parent_molecules_count": len(df_parents),
    "total_transforms_attempted": len(generated_records) + len(exclusion_records),
    "total_unique_valid_analogues": len(df_pred),
    "total_exclusions": len(df_excl),
    "track_a_analogues_count": int((df_pred["track"] == "Track A").sum()),
    "track_b_analogues_count": int((df_pred["track"] == "Track B").sum()),
    "passes_pIC50_count": int(df_pred["passes_pIC50_threshold"].sum()),
    "passes_logBB_count": int(df_pred["passes_logBB_threshold"].sum()),
    "passes_both_joint_hits_count": len(joint_hits),
    "clean_joint_hits_count": int(sum(h["chemically_clean"] for h in joint_hits)),
    "non_duplicate_joint_hits_count": int(sum(not h["exact_training_duplicate"] for h in joint_hits)),
    "joint_hits": joint_hits,
}

with open(OUT_DIR / "optimization_summary.json", "w", encoding="utf-8") as f:
    json.dump(opt_summary, f, indent=2)

print(f"\nAll deliverables written to {OUT_DIR}/")
