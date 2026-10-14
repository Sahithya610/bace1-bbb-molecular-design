"""Final check: print docked + validated rows from master table."""
import pandas as pd
import os
from pathlib import Path

base = str(Path(__file__).resolve().parent.parent)
out_path = os.path.join(base, 'scripts', 'phase2_final_check_out.txt')

with open(out_path, 'w', encoding='utf-8') as f:
    def p(s=''):
        print(s)
        f.write(s + '\n')

    master = pd.read_csv(os.path.join(base, 'results', 'bace1_master_candidate_table.csv'))

    p("=== DOCKED ROWS (passes_BBB_and_docking not NA, showing key cols) ===")
    docked = master[master['docking_status'].notna()][
        ['canonical_smiles','oracle_predicted_pIC50','oracle_predicted_logBB',
         'vina_score','passes_BBB_and_docking','chem_verdict']
    ].sort_values('vina_score')
    for _, r in docked.iterrows():
        p(f"  pIC50={r['oracle_predicted_pIC50']:.4f}  logBB={r['oracle_predicted_logBB']:.4f}  "
          f"vina={r['vina_score']}  BBB+dock={r['passes_BBB_and_docking']}  "
          f"verdict={r['chem_verdict']}  "
          f"smi={str(r['canonical_smiles'])[:60]}")

    p()
    p("=== CHEMISTRY-FILTER ROWS ===")
    chemrows = master[master['chem_verdict'].notna()][
        ['canonical_smiles','oracle_predicted_pIC50','oracle_predicted_logBB',
         'chem_verdict','passes_all_chemistry_filters','passes_BBB_and_docking']
    ]
    for _, r in chemrows.iterrows():
        p(f"  verdict={r['chem_verdict']}  passes_all={r['passes_all_chemistry_filters']}  "
          f"pIC50={r['oracle_predicted_pIC50']:.4f}  logBB={r['oracle_predicted_logBB']:.4f}  "
          f"BBB+dock={r['passes_BBB_and_docking']}")
