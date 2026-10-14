# DrugGEN–BACE1 Executive Summary

**Project:** Extension of DrugGEN (Nature Machine Intelligence 2025) to BACE1 Alzheimer's target with BBB permeability objective  
**Status:** Computational study — no experimental validation performed  
**Date:** October 2026

---

## Objective

Generate BACE1 inhibitor candidates satisfying **both**:
- Predicted pIC50 ≥ 7.6576 (oracle ExtraTrees model, scaffold-split test R² = 0.616, RMSE = 0.755)
- Predicted logBB ≥ 0.0 (BBB permeability model, scaffold-split test R² = 0.354, RMSE = 0.621)

---

## Approach

1. Extended DrugGEN graph-transformer GAN to BACE1 with dual conditioning (pIC50, logBB)
2. Tested **8 conditioning approaches** to make generation respond to requested pIC50 — all failed (Δ pIC50 < 0.3 or validity collapse)
3. Fell back to **post-hoc oracle filtering**: generated 16,384 molecules, oracle-ranked, docked top 50 candidates
4. Applied medicinal chemistry filters (PAINS, Brenk, reactive groups)

---

## Key Results

| Outcome Category | Count |
|-----------------|-------|
| **A. Molecules meeting BOTH original thresholds** | **0** |
| B. BBB-positive AND Vina ≤ −7.5 kcal/mol | 11 |
| C. CLEAN (no PAINS/Brenk alerts) among B | 2 |
| D. High oracle pIC50 (≥7.0), fail BBB | 5 |

**Maximum predicted pIC50:** 7.904 — predicted logBB = −0.020 (BBB-negative; fails joint threshold)

**Best fully clean candidate:** predicted pIC50 = 6.169, predicted logBB = +0.023, Vina = −8.03 kcal/mol, QED = 0.643, MW = 426 — does NOT meet pIC50 ≥ 7.6576 threshold

---

## Honest Limitations

- **Zero molecules** meet the original joint computational threshold
- Oracle typical error: RMSE = 0.755 pIC50 units (test R² = 0.616)
- BBB model typical error: RMSE = 0.621 log units (test R² = 0.354); BACE1 scaffolds are outside B3DB training domain
- Docking used simplified receptor PDBQT; scores are for relative ranking only
- All results are **in silico predictions** — no synthesis or wet-lab validation

---

## Recommended Next Step

1. **Experimental:** Synthesise and test the 2 fully clean candidates (cheapest path to any experimental result)
2. **Computational:** Fix BBB applicability domain gap; improve receptor preparation (H++ protonation of D32/D228) before further docking
3. **Architecture:** Implement and test cross-attention conditioning as the next generation experiment (see full report Phase 6)

---

*All values are computational predictions. None constitute measured activity, confirmed brain penetration, or validated drug candidates.*
