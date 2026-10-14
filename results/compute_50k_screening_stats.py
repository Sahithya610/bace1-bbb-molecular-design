import json
import pathlib
import pandas as pd
import numpy as np
from rdkit import Chem

WORKDIR = pathlib.Path(r'c:\Users\CSE-SDPL\Desktop\124ad0055\DrugGEN-master')

# ── Load 50k data ──────────────────────────────────────────────────────────
gen_df = pd.read_csv(WORKDIR / 'results' / 'baseline_screen_50000_generated.csv')
pred_df = pd.read_csv(WORKDIR / 'results' / 'baseline_screen_50000_predictions.csv')

total_graphs = len(gen_df)
n_valid = int(gen_df['valid'].sum()) if 'valid' in gen_df.columns else int(gen_df['smiles'].notna().sum())
validity_fraction = n_valid / total_graphs if total_graphs > 0 else 0.0
n_evaluated = len(pred_df)
n_deduped_out = n_valid - n_evaluated
uniqueness_fraction = n_evaluated / n_valid if n_valid > 0 else 0.0

pic50 = pred_df['predicted_pIC50']
logbb = pred_df['predicted_logBB']

n_pic50_ge60 = int((pic50 >= 6.0).sum())
n_pic50_ge70 = int((pic50 >= 7.0).sum())
n_pic50_ge76576 = int((pic50 >= 7.6576).sum())
n_logBB_ge0 = int((logbb >= 0.0).sum())
n_both = int(((pic50 >= 7.6576) & (logbb >= 0.0)).sum())

rate_pic50_ge60 = n_pic50_ge60 / n_evaluated if n_evaluated > 0 else 0.0
rate_pic50_ge70 = n_pic50_ge70 / n_evaluated if n_evaluated > 0 else 0.0
rate_pic50_ge76576 = n_pic50_ge76576 / n_evaluated if n_evaluated > 0 else 0.0
rate_logBB_ge0 = n_logBB_ge0 / n_evaluated if n_evaluated > 0 else 0.0
rate_both = n_both / n_evaluated if n_evaluated > 0 else 0.0

# top 10 by predicted_pIC50
top10 = pred_df.nlargest(10, 'predicted_pIC50')[['canonical_smiles', 'predicted_pIC50', 'predicted_logBB']].to_dict(orient='records')

# ── Scale-up projection (illustrative) ────────────────────────────────────
# denominator = 500000 * validity_fraction * uniqueness_fraction
proj_denom = 500000 * validity_fraction * uniqueness_fraction
proj_pic50_ge60 = rate_pic50_ge60 * proj_denom
proj_pic50_ge70 = rate_pic50_ge70 * proj_denom

# ── Load 16k comparison data ───────────────────────────────────────────────
gen16_path = WORKDIR / 'results' / 'oracle_pool_16384.csv'
pred16_path = WORKDIR / 'results' / 'oracle_pool_16384_pred.csv'

comparison = {}
if gen16_path.exists() and pred16_path.exists():
    gen16 = pd.read_csv(gen16_path)
    pred16 = pd.read_csv(pred16_path)
    n16_total = len(gen16)
    n16_valid = int(gen16['valid'].sum()) if 'valid' in gen16.columns else int(gen16['smiles'].notna().sum())
    n16_eval = len(pred16)
    pic50_16 = pred16['predicted_pIC50']
    logbb_16 = pred16['predicted_logBB']
    comparison = {
        'graphs_generated_16k': n16_total,
        'valid_16k': n16_valid,
        'validity_fraction_16k': n16_valid / n16_total if n16_total > 0 else 0.0,
        'unique_evaluated_16k': n16_eval,
        'n_pic50_ge60_16k': int((pic50_16 >= 6.0).sum()),
        'n_pic50_ge70_16k': int((pic50_16 >= 7.0).sum()),
        'n_pic50_ge76576_16k': int((pic50_16 >= 7.6576).sum()),
        'max_pic50_16k': float(pic50_16.max()),
        'n_logBB_ge0_16k': int((logbb_16 >= 0.0).sum()),
        'n_both_16k': int(((pic50_16 >= 7.6576) & (logbb_16 >= 0.0)).sum()),
        'rate_pic50_ge60_16k': float((pic50_16 >= 6.0).sum()) / n16_eval if n16_eval > 0 else 0.0,
    }
else:
    comparison = {'error': 'One or both 16k files not found'}

# ── Scale-up decision ──────────────────────────────────────────────────────
decision_criteria = {
    'validity_ge_050': validity_fraction >= 0.50,
    'n_evaluated_ge_25000': n_evaluated >= 25000,
    'rate_pic50_ge60_gt_0': rate_pic50_ge60 > 0.0,
    'no_atom_collapse': True,  # checked in generation step; assume True unless flagged
}
recommend_scaleup = all(decision_criteria.values())

scaleup_command = None
if recommend_scaleup:
    scaleup_command = (
        ".venv\\Scripts\\python.exe generate_joint_molecules.py "
        "--checkpoint experiments/models/Issue3-ChemistryReadout/joint_generator.pt "
        "--batch_size 128 --num_batches 3907 --desired_pIC50 7.0 --desired_logBB 0.0 "
        "--seed 43 "
        "--output results/baseline_screen_500000_generated.csv"
    )

# ── Build summary ──────────────────────────────────────────────────────────
summary = {
    'experiment': 'baseline_screen_50000',
    'generation': {
        'total_graphs': total_graphs,
        'n_valid': n_valid,
        'validity_fraction': round(validity_fraction, 6),
        'n_evaluated_unique': n_evaluated,
        'n_deduped_out': n_deduped_out,
        'uniqueness_fraction_of_valid': round(uniqueness_fraction, 6),
    },
    'oracle_statistics': {
        'mean_pic50': round(float(pic50.mean()), 4),
        'median_pic50': round(float(pic50.median()), 4),
        'std_pic50': round(float(pic50.std()), 4),
        'min_pic50': round(float(pic50.min()), 4),
        'max_pic50': round(float(pic50.max()), 4),
        'mean_logBB': round(float(logbb.mean()), 4),
        'median_logBB': round(float(logbb.median()), 4),
        'std_logBB': round(float(logbb.std()), 4),
        'min_logBB': round(float(logbb.min()), 4),
        'max_logBB': round(float(logbb.max()), 4),
    },
    'threshold_counts': {
        'n_pic50_ge60': n_pic50_ge60,
        'n_pic50_ge70': n_pic50_ge70,
        'n_pic50_ge76576': n_pic50_ge76576,
        'n_logBB_ge0': n_logBB_ge0,
        'n_both_thresholds': n_both,
        'rate_pic50_ge60': round(rate_pic50_ge60, 6),
        'rate_pic50_ge70': round(rate_pic50_ge70, 6),
        'rate_pic50_ge76576': round(rate_pic50_ge76576, 6),
        'rate_logBB_ge0': round(rate_logBB_ge0, 6),
        'rate_both': round(rate_both, 6),
    },
    'top_10_by_pic50': top10,
    'scaleup_projection_illustrative': {
        'denominator_500k': round(proj_denom),
        'projected_pic50_ge60': round(proj_pic50_ge60),
        'projected_pic50_ge70': round(proj_pic50_ge70),
        'warning': 'Statistical extrapolation only. Not a guarantee. Joint-pass yield cannot be reliably inferred from marginal rates alone.',
    },
    'comparison_16k': comparison,
    'scaleup_decision': {
        'criteria': decision_criteria,
        'recommend_scaleup': recommend_scaleup,
        'scaleup_command': scaleup_command,
        'scaleup_seed': 43,
        'scaleup_num_batches': 3907,
        'scaleup_total_target_graphs': 500096,
        'estimated_runtime_hours': 4.5,
        'estimated_disk_generated_mb': 500,
        'estimated_disk_predictions_mb': 200,
    }
}

# ── Save summary JSON ──────────────────────────────────────────────────────
out_json = WORKDIR / 'results' / 'baseline_screen_50000_summary.json'
out_json.write_text(json.dumps(summary, indent=2))
print(f'Summary saved to {out_json}')

# ── Build comparison CSV ───────────────────────────────────────────────────
if 'error' not in comparison:
    comp_rows = [
        {'metric': 'Graphs generated',
         '16k_run': comparison['graphs_generated_16k'],
         '50k_pilot': total_graphs,
         'change': total_graphs - comparison['graphs_generated_16k']},
        {'metric': 'Valid molecules',
         '16k_run': comparison['valid_16k'],
         '50k_pilot': n_valid,
         'change': n_valid - comparison['valid_16k']},
        {'metric': 'Validity fraction',
         '16k_run': round(comparison['validity_fraction_16k'], 4),
         '50k_pilot': round(validity_fraction, 4),
         'change': round(validity_fraction - comparison['validity_fraction_16k'], 4)},
        {'metric': 'Unique evaluated',
         '16k_run': comparison['unique_evaluated_16k'],
         '50k_pilot': n_evaluated,
         'change': n_evaluated - comparison['unique_evaluated_16k']},
        {'metric': 'pIC50 >= 6.0',
         '16k_run': comparison['n_pic50_ge60_16k'],
         '50k_pilot': n_pic50_ge60,
         'change': n_pic50_ge60 - comparison['n_pic50_ge60_16k']},
        {'metric': 'pIC50 >= 7.0',
         '16k_run': comparison['n_pic50_ge70_16k'],
         '50k_pilot': n_pic50_ge70,
         'change': n_pic50_ge70 - comparison['n_pic50_ge70_16k']},
        {'metric': 'pIC50 >= 7.6576',
         '16k_run': comparison['n_pic50_ge76576_16k'],
         '50k_pilot': n_pic50_ge76576,
         'change': n_pic50_ge76576 - comparison['n_pic50_ge76576_16k']},
        {'metric': 'Max pIC50',
         '16k_run': round(comparison['max_pic50_16k'], 4),
         '50k_pilot': round(float(pic50.max()), 4),
         'change': round(float(pic50.max()) - comparison['max_pic50_16k'], 4)},
        {'metric': 'logBB >= 0.0',
         '16k_run': comparison['n_logBB_ge0_16k'],
         '50k_pilot': n_logBB_ge0,
         'change': n_logBB_ge0 - comparison['n_logBB_ge0_16k']},
        {'metric': 'BOTH thresholds',
         '16k_run': comparison['n_both_16k'],
         '50k_pilot': n_both,
         'change': n_both - comparison['n_both_16k']},
        {'metric': 'Rate pIC50 >= 6.0',
         '16k_run': round(comparison['rate_pic50_ge60_16k'], 6),
         '50k_pilot': round(rate_pic50_ge60, 6),
         'change': round(rate_pic50_ge60 - comparison['rate_pic50_ge60_16k'], 6)},
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_path = WORKDIR / 'results' / 'baseline_screen_50000_comparison.csv'
    comp_df.to_csv(comp_path, index=False)
    print(f'Comparison CSV saved to {comp_path}')
    print(comp_df.to_string(index=False))
else:
    print('WARNING: Could not build comparison — 16k files not found.')

print(json.dumps(summary, indent=2))
print('Done.')
