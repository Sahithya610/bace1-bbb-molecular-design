"""
Compute pilot metrics for the cross-attention conditioning pilot (Phase 4).

Reads:
    results/cross_attention_pilot/pred_cond55.csv
    results/cross_attention_pilot/pred_cond80.csv

Computes:
    delta_pic50    = mean(oracle@8.0) - mean(oracle@5.5)  [non-trivial only]
    valid_frac_55  = count(valid AND len(smiles)>3) / 1024
    valid_frac_80  = count(valid AND len(smiles)>3) / 1024
    n_lo           = non-trivial count in pred_cond55
    n_hi           = non-trivial count in pred_cond80

PASS: delta_pic50>=0.3 AND valid_frac_55>=0.50 AND valid_frac_80>=0.50
      AND n_lo>=10 AND n_hi>=10

Saves:
    results/cross_attention_pilot/pilot_results.json
    reports/cross_attention_decision.md
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

TOTAL = 1024

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def load_pred(path):
    df = pd.read_csv(path)
    return df


def non_trivial(df):
    """Rows where valid=True AND len(smiles)>3."""
    if "valid" in df.columns:
        mask_valid = df["valid"].astype(bool)
    else:
        # predict_joint.py output: canonical_smiles column present means valid
        mask_valid = df["canonical_smiles"].notna() & (df["canonical_smiles"].str.len() > 0)
    mask_len = df["smiles"].apply(lambda s: isinstance(s, str) and len(s) > 3)
    return df[mask_valid & mask_len].copy()


def main():
    root = Path("results/cross_attention_pilot")
    p55 = root / "pred_cond55.csv"
    p80 = root / "pred_cond80.csv"

    # Also try raw gen files for valid_frac (gen has 'valid' column)
    g55 = root / "gen_cond55.csv"
    g80 = root / "gen_cond80.csv"

    print("Loading prediction CSVs...")
    df55 = load_pred(p55)
    df80 = load_pred(p80)

    print(f"pred_cond55 rows: {len(df55)}")
    print(f"pred_cond80 rows: {len(df80)}")
    print(f"Columns 55: {list(df55.columns)}")
    print(f"Columns 80: {list(df80.columns)}")

    # ---------------------------------------------------------------
    # valid_frac from gen csv (has valid column + all 1024 rows)
    # ---------------------------------------------------------------
    if g55.exists() and g80.exists():
        gen55 = pd.read_csv(g55)
        gen80 = pd.read_csv(g80)
        nt55_gen = gen55[gen55["valid"].astype(bool) & gen55["smiles"].apply(lambda s: isinstance(s, str) and len(s) > 3)]
        nt80_gen = gen80[gen80["valid"].astype(bool) & gen80["smiles"].apply(lambda s: isinstance(s, str) and len(s) > 3)]
        valid_frac_55 = len(nt55_gen) / TOTAL
        valid_frac_80 = len(nt80_gen) / TOTAL
        print(f"Non-trivial from gen: cond55={len(nt55_gen)}, cond80={len(nt80_gen)}")
    else:
        # Fallback: use prediction file
        nt55_gen = non_trivial(df55)
        nt80_gen = non_trivial(df80)
        valid_frac_55 = len(nt55_gen) / TOTAL
        valid_frac_80 = len(nt80_gen) / TOTAL

    # ---------------------------------------------------------------
    # non-trivial from prediction files
    # ---------------------------------------------------------------
    nt55 = non_trivial(df55)
    nt80 = non_trivial(df80)
    n_lo = len(nt55)
    n_hi = len(nt80)

    print(f"\nNon-trivial predictions: cond55={n_lo}, cond80={n_hi}")

    # ---------------------------------------------------------------
    # Oracle pIC50
    # ---------------------------------------------------------------
    pic50_col = "predicted_pIC50"
    if pic50_col not in df55.columns:
        raise KeyError(f"Column '{pic50_col}' not found in pred_cond55. Cols: {list(df55.columns)}")

    if n_lo > 0:
        mean_oracle_55 = float(nt55[pic50_col].mean())
    else:
        mean_oracle_55 = float(df55[pic50_col].mean()) if len(df55) > 0 else float("nan")

    if n_hi > 0:
        mean_oracle_80 = float(nt80[pic50_col].mean())
    else:
        mean_oracle_80 = float(df80[pic50_col].mean()) if len(df80) > 0 else float("nan")

    delta_pic50 = mean_oracle_80 - mean_oracle_55

    print(f"mean_oracle_55 = {mean_oracle_55:.4f}")
    print(f"mean_oracle_80 = {mean_oracle_80:.4f}")
    print(f"delta_pic50    = {delta_pic50:.4f}")
    print(f"valid_frac_55  = {valid_frac_55:.4f}")
    print(f"valid_frac_80  = {valid_frac_80:.4f}")
    print(f"n_lo           = {n_lo}")
    print(f"n_hi           = {n_hi}")

    # ---------------------------------------------------------------
    # PASS criteria
    # ---------------------------------------------------------------
    passed = (
        delta_pic50  >= 0.3  and
        valid_frac_55 >= 0.50 and
        valid_frac_80 >= 0.50 and
        n_lo >= 10            and
        n_hi >= 10
    )

    conclusion_parts = []
    if delta_pic50 < 0.3:
        conclusion_parts.append(f"delta_pic50={delta_pic50:.4f} < 0.3 FAIL")
    else:
        conclusion_parts.append(f"delta_pic50={delta_pic50:.4f} >= 0.3 PASS")
    if valid_frac_55 < 0.50:
        conclusion_parts.append(f"valid_frac_55={valid_frac_55:.4f} < 0.50 FAIL")
    else:
        conclusion_parts.append(f"valid_frac_55={valid_frac_55:.4f} >= 0.50 PASS")
    if valid_frac_80 < 0.50:
        conclusion_parts.append(f"valid_frac_80={valid_frac_80:.4f} < 0.50 FAIL")
    else:
        conclusion_parts.append(f"valid_frac_80={valid_frac_80:.4f} >= 0.50 PASS")
    if n_lo < 10:
        conclusion_parts.append(f"n_lo={n_lo} < 10 FAIL")
    else:
        conclusion_parts.append(f"n_lo={n_lo} >= 10 PASS")
    if n_hi < 10:
        conclusion_parts.append(f"n_hi={n_hi} < 10 FAIL")
    else:
        conclusion_parts.append(f"n_hi={n_hi} >= 10 PASS")

    conclusion = ("PASS — cross-attention conditioning works. " if passed else "FAIL — ") + "; ".join(conclusion_parts)

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    results = {
        "delta_pic50":    round(delta_pic50, 6),
        "valid_frac_55":  round(valid_frac_55, 6),
        "valid_frac_80":  round(valid_frac_80, 6),
        "n_lo":           int(n_lo),
        "n_hi":           int(n_hi),
        "mean_oracle_55": round(mean_oracle_55, 6),
        "mean_oracle_80": round(mean_oracle_80, 6),
        "pass":           passed,
        "conclusion":     conclusion,
    }

    out_json = root / "pilot_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_json}")

    # ---------------------------------------------------------------
    # Decision report
    # ---------------------------------------------------------------
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    status = "PASS" if passed else "FAIL"

    md = f"""# Cross-Attention Conditioning Decision Report

## Decision Rationale

**YES — cross-attention was warranted because additive embedding causes saturation.**

In all 8 prior approaches the condition was injected additively:
`node = node + c.unsqueeze(1)` adds the same condition vector to ALL 45 atom positions.
At high pIC50 requests (e.g. 8.0) this saturates atom embeddings uniformly so the
generator has no way to specialize atom-level structure to satisfy the condition.

Cross-attention lets each atom position *query* the condition token independently,
learning which positions should respond to the pIC50/logBB signal.

## Implementation Description

- **Architecture**: `CrossAttentionGenerator` subclasses `ConditionalGenerator`.
- **Location**: after `TransformerEncoder`, before `readout_condition` concatenation.
- **New layers**:
  - `cross_cond_proj = Linear(condition_dim=3, dim=128)` → condition token [B, 1, 128]
  - `cross_attn = MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)`
- **Residual**: `node = node + cross_attn(query=node, key/value=condition_token)`
- **Existing paths unchanged**: `condition_layers` (Transformer-level), `readout_condition` (chemistry-level).
- **Warm-start**: `strict=False` from `Issue3-ChemistryReadout`; new params initialized fresh.
- **Training**: 500 steps, batch=32, lambda_gp=10, lambda_aux=1, lambda_condition=5.

## Pilot Result: {status}

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| delta_pic50 | {delta_pic50:.4f} | ≥ 0.3 | {"✅" if delta_pic50 >= 0.3 else "❌"} |
| valid_frac_55 | {valid_frac_55:.4f} | ≥ 0.50 | {"✅" if valid_frac_55 >= 0.50 else "❌"} |
| valid_frac_80 | {valid_frac_80:.4f} | ≥ 0.50 | {"✅" if valid_frac_80 >= 0.50 else "❌"} |
| n_lo | {n_lo} | ≥ 10 | {"✅" if n_lo >= 10 else "❌"} |
| n_hi | {n_hi} | ≥ 10 | {"✅" if n_hi >= 10 else "❌"} |

**mean_oracle_55** = {mean_oracle_55:.4f}
**mean_oracle_80** = {mean_oracle_80:.4f}
**delta_pic50**    = {delta_pic50:.4f}

**Conclusion**: {conclusion}

## Comparison to Prior 8 Approaches

| Approach | Best delta_pic50 | Notes |
|----------|-----------------|-------|
| Lambda sweep (lambda=1) | +0.066 | n_lo=189, n_hi=305; saturation at lambda>1 |
| Lambda sweep (lambda>1) | negative/zero | generator collapse |
| V-gated sweep (all) | near zero | same saturation root cause |
| REINFORCE pilot v4 | near zero | reward shaping had no gradient path |
| ET-distill baseline | ~0.010 | head alignment better but still additive injection |
| All 8 approaches combined | best +0.066 | all used additive injection |

The cross-attention pilot achieved **delta_pic50 = {delta_pic50:.4f}** vs the best prior +0.066.

## Recommendation for Next Step

{"**Full cross-attention training**: Run 10 epochs (full dataset) with the same hyperparameters. The 500-step pilot demonstrated the gradient signal is present; more training should strengthen it." if passed else "**Investigate**: delta_pic50=" + f"{delta_pic50:.4f}" + " is not yet at the 0.3 threshold. Consider: (1) increase lambda_condition further (e.g. 10.0), (2) run more steps (2000+), (3) add cross-attention to edge features as well, (4) verify the warm-start did not suppress the cross-attn gradient by checking the norm of cross_cond_proj gradients."}

---
*Generated by compute_ca_pilot_results.py — Issue 4 Cross-Attention Pilot*
"""

    out_md = report_dir / "cross_attention_decision.md"
    out_md.write_text(md)
    print(f"Saved: {out_md}")

    print("\n" + "=" * 70)
    print(f"PILOT RESULT: {status}")
    print(f"delta_pic50 = {delta_pic50:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
