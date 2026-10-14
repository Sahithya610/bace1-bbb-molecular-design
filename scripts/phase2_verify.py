"""Quick verification of the Phase 2 output files."""
import pandas as pd
import os
from pathlib import Path

base = str(Path(__file__).resolve().parent.parent)
out_path = os.path.join(base, 'scripts', 'phase2_verify_out.txt')

with open(out_path, 'w', encoding='utf-8') as f:
    def p(s=''):
        print(s)
        f.write(s + '\n')

    # Check master table
    master_path = os.path.join(base, 'results', 'bace1_master_candidate_table.csv')
    p(f"Reading: {master_path}")
    master = pd.read_csv(master_path)
    p(f"Shape: {master.shape}")
    p(f"Columns: {list(master.columns)}")
    p()

    total = len(master)
    p(f"Total rows: {total:,}  (expected 11,376)")

    n_pic50 = master['passes_pIC50_7_6576'].sum()
    p(f"passes_pIC50_7_6576: {n_pic50:,} / {total:,}  ({n_pic50/total:.1%})")

    n_logbb = master['passes_logBB_0'].sum()
    p(f"passes_logBB_0: {n_logbb:,} / {total:,}  ({n_logbb/total:.1%})")

    n_joint = master['passes_original_joint_threshold'].sum()
    p(f"passes_original_joint_threshold (BOTH): {n_joint:,}  (expected ZERO)")

    n_bbb_dock = (master['passes_BBB_and_docking'] == True).sum()
    p(f"passes_BBB_and_docking = True: {n_bbb_dock:,}  (expected 11)")

    n_clean = (master['passes_all_chemistry_filters'] == True).sum()
    p(f"passes_all_chemistry_filters = True: {n_clean:,}  (expected 2)")

    n_high_fail = ((master['oracle_predicted_pIC50'] >= 7.0) & (master['oracle_predicted_logBB'] < 0.0)).sum()
    p(f"High pIC50 (>=7.0) failing BBB: {n_high_fail:,}  (expected ~5)")

    p()
    p("=== KEY CANDIDATE VERIFICATION ===")
    from rdkit import Chem
    def can(s):
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol) if mol else None

    for label, smi, exp_pic50, exp_logBB in [
        ("Candidate 1 (pIC50≈7.9037, logBB≈-0.020)",
         "CC(C)(C)COc1cnc2c(c1)C1(COC(N)=N1)c1cc(-c3cncnc3)ccc1O2", 7.9037, -0.020),
        ("Candidate 2 (pIC50≈6.169, logBB≈+0.023)",
         "CC1CCC(COc2ccc(F)cn2)CN1C(O)c1cc(F)ccc1-c1ncccn1", 6.169, 0.023),
    ]:
        canon = can(smi)
        row = master[master['canonical_smiles'] == canon]
        p(f"\n{label}")
        if len(row) == 0:
            p(f"  NOT FOUND (tried: {canon})")
        else:
            r = row.iloc[0]
            p(f"  pIC50 = {r['oracle_predicted_pIC50']:.6f}  (expected ≈{exp_pic50})")
            p(f"  logBB = {r['oracle_predicted_logBB']:.6f}  (expected ≈{exp_logBB})")
            if pd.notna(r.get('vina_score')):
                p(f"  vina_score = {r['vina_score']}")
            if pd.notna(r.get('chem_verdict')):
                p(f"  chem_verdict = {r['chem_verdict']}")
            if pd.notna(r.get('passes_BBB_and_docking')):
                p(f"  passes_BBB_and_docking = {r['passes_BBB_and_docking']}")
            if pd.notna(r.get('max_training_Tc')):
                p(f"  max_training_Tc = {r['max_training_Tc']:.4f}")
                p(f"  nearest_training_smiles = {r['nearest_training_smiles']}")
                p(f"  exact_training_duplicate = {r['exact_training_duplicate']}")
            else:
                p("  max_training_Tc = NaN  (Tanimoto not computed for this row)")

    p()
    p("=== TOP-200 TANIMOTO STATUS ===")
    top200 = master[master['t200_scaffold'].notna()]
    p(f"Top-200 rows: {len(top200)}")
    tc_notna = top200['max_training_Tc'].notna().sum()
    p(f"max_training_Tc populated: {tc_notna}")
    if tc_notna > 0:
        p(f"max_training_Tc range: {top200['max_training_Tc'].dropna().min():.4f} - {top200['max_training_Tc'].dropna().max():.4f}")
        n_exact = (top200['exact_training_duplicate'] == True).sum()
        p(f"exact_training_duplicate = True: {n_exact}")

    p()
    p("=== EXCLUSIONS TABLE ===")
    excl_path = os.path.join(base, 'results', 'bace1_candidate_exclusions.csv')
    excl = pd.read_csv(excl_path)
    p(f"Shape: {excl.shape}")
    p(f"Columns: {list(excl.columns)}")
    p(f"exclusion_category value counts:\n{excl['exclusion_category'].value_counts().to_string()}")

    p()
    p("=== PHASE 2 COMPLETE ===")
    p(f"Master table: {master_path}  ({total:,} rows × {len(master.columns)} cols)")
    p(f"Exclusions: {excl_path}  ({len(excl):,} rows)")
