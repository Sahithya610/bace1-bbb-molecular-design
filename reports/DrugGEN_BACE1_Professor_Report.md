# DrugGEN–BACE1 Professor Report

**Author:** Computational Drug Discovery Study  
**Status:** Computational only — no experimental validation  
**Date:** October 2026  
**Base paper:** Ozcelik et al., *Nature Machine Intelligence* 2025, DOI: 10.1038/s42256-025-01082-y

> **Terminology note.** Throughout this report: "predicted pIC50" refers to ExtraTrees oracle model output; "predicted logBB" refers to a separate B3DB-trained regression model output; "docking score" refers to AutoDock Vina v1.2.7 empirical scoring function output. None of these constitute experimentally measured values. No synthesis or in-vitro validation has been performed.

---

## 1. Research Question and Motivation

Beta-secretase 1 (BACE1) is a validated target for Alzheimer's disease. Inhibiting BACE1 reduces amyloid-beta production, a central hallmark of the disease. A key challenge is that effective BACE1 inhibitors must also penetrate the blood-brain barrier (BBB) to reach the CNS target.

**Research question:** Can the DrugGEN graph-generative framework be extended to produce novel BACE1 inhibitor candidates that simultaneously satisfy predicted activity (pIC50 ≥ 7.6576) and predicted CNS accessibility (logBB ≥ 0.0)?

**Original thresholds (unchanged throughout):**
- Oracle-predicted pIC50 ≥ 7.6576
- Predicted logBB ≥ 0.0

---

## 2. Base Paper and Extension

**Base paper:** "Target-specific de novo design of drug candidate molecules with graph-transformer-based generative adversarial networks" (Ozcelik et al., *Nature Machine Intelligence* 2025).

The original DrugGEN framework uses a graph-transformer GAN architecture to generate drug-like molecules targeting specific proteins. It was validated on AKT1 and CDK2.

**This project's extension:**
- Changed target from AKT1/CDK2 to BACE1
- Added continuous pIC50 and logBB conditioning (BACE1 activity + BBB permeability)
- Trained ExtraTrees oracle model on 8,272 BACE1 ChEMBL ligands
- Trained B3DB BBB regression model on 1,058 experimental logBB values
- Systematically tested 8 conditioning strategies

---

## 3. Dataset and Preprocessing

**BACE1 activity data:**
- Source: ChEMBL database, 8,272 unique canonical SMILES
- pIC50 definition: pIC50 = 9 − log10(IC50 [nM])
- pIC50 range: 1.432 – 12.000 (mean 6.677, median 6.796)
- 44.4% of training molecules have pIC50 ≥ 7.0; 15.9% have pIC50 ≥ 8.0
- Split: scaffold-based, 5,789 train / 1,242 val / 1,240 test (2,082/404/439 unique scaffolds)
- No duplicate canonical SMILES in training set

**BBB permeability data:**
- Source: B3DB quantitative regression dataset, 1,058 molecules with experimental logBB values
- B3DB logBB range approximately −3 to +1 (CNS drugs and probe compounds)
- Split: scaffold-based, 742 train / 177 val / 139 test
- **Critical note:** The `predicted_logBB` column in `bace1_continuous_bbb.csv` contains model predictions, NOT experimental values. The BBB model was applied to BACE1 molecules (outside B3DB training domain). BBB scores for generated BACE1 candidates are therefore predictions-of-predictions with unverified applicability domain.

**Condition normalization:**
- pIC50: mean = 6.691, std = 1.309
- logBB: mean = −0.078, std = 0.752

---

## 4. Architecture Overview

**ConditionalGenerator** (extended from DrugGEN):
- Graph Transformer, condition_dim = 3: [BACE1=1.0, normalized_pIC50, normalized_logBB]
- Condition injection: additive embedding at (1) input node/edge projections and (2) immediately before categorical readout layers
- Output: discrete atom-type logits [B, V=45, 14] and bond-type logits [B, V, V, 5]

**ConditionalDiscriminator:**
- WGAN-GP critic for adversarial loss
- Separate condition-free property head (graph-aware, supervised by oracle predictions)

**Key architectural constraint:** The condition is added uniformly to all 45 node positions. A high pIC50 request (normalized ≈ +1.0) biases all positions simultaneously away from the padding class (atom class 0), creating saturation pressure that causes validity collapse at high λ_condition values.

---

## 5. Conditioning Experiments (All 8 Approaches)

**Measurement protocol:** Δ pIC50 = mean(oracle predictions | requested pIC50 = 8.0) − mean(oracle predictions | requested pIC50 = 5.5), with requested logBB = 0.0.  
**Pass criterion:** |Δ pIC50| ≥ 0.3 AND ValidFrac ≥ 0.50.

| # | Approach | Hypothesis | Steps | Δ pIC50 | ValidFrac | Collapse mode | Key log |
|---|----------|-----------|-------|---------|-----------|---------------|---------|
| 1 | Baseline chemistry-readout head | Internal head aligns proxy to oracle | 10 epochs | +0.066 (Δ at 500 steps) | ~80% | Head/oracle feature-space mismatch; generator never learns BACE1 pharmacophore | `et_distill_pilot500_log.txt` |
| 2 | ET-distill head supervision | Supervise head with ExtraTrees predictions to bridge spaces | 500 | +0.066 | ~80% | Same external gap; head learns ET-latent features, not pharmacophore | `et_distill_pilot500_log.txt` |
| 3 | λ_condition sweep (λ=1,5,10,20) | Amplify condition signal weight | 500 each | +0.066 @ λ=1; 0 valid @ 8.0 for λ≥5 | Collapses at λ≥5 | All-45-atom saturation: high λ pushes every position to non-padding | `lambda_sweep_results.txt` |
| 4 | Validity-gated condition loss | Gate cond_loss to valid molecules only | 500 | 0/1024 valid at pIC50=8.0 | ValidFrac=0.984 during training | Gate fixes training statistics; generation still collapses (model learned to saturate at pIC50=8.0) | `vgated_sweep_results.txt` |
| 5 | Padding penalty (λ_c=5, λ_p=5) | Reward padding probability to counteract saturation | 500 | Δ=+0.529 (SPURIOUS: n_lo=2, both trivial single-atom molecules) | 1020/1024 "valid" but all SMILES len≤3 | Degenerates to C/F single atoms; padding penalty forces minimal graphs | `vgated_sweep2_results.txt` |
| 6 | REINFORCE raw oracle reward | Direct oracle signal to generator | 500 | corr = −0.155 | 63% | Reward not conditioned on request; generator maximises average score | `reinforce_pilot_log.txt` |
| 7 | REINFORCE conditional exp(−\|Δ\|/τ) | Soft match reward, always positive | 500 | corr = −0.428 | 29% | Always-positive reward equivalent to entropy maximisation; worse than baseline | `reinforce_pilot_v3_log.txt` |
| 8 | Contrastive batch REINFORCE | Within-batch contrast (16 low + 16 high pIC50) | 500 | corr = −0.653 | 36% | High-dimensional action space; within-batch contrast anti-correlated | `reinforce_pilot_v4_log.txt` |

**Result: All 8 approaches failed. No configuration achieved |Δ pIC50| ≥ 0.3 with ValidFrac ≥ 0.50 simultaneously.**

---

## 6. Root Cause Analysis

The persistent failure has multiple compounding causes:

**A. Additive embedding saturation**  
The condition is added uniformly to all 45 atom position embeddings. A normalized pIC50 of +1.0 (≈ requested pIC50 = 8.0) biases all positions simultaneously away from atom class 0 (padding). At high λ_condition, this causes atom-slot saturation: all 45 positions fill with non-padding atoms, producing invalid 45-heavy-atom graphs that cannot decode to valid SMILES.

**B. Proxy head / oracle feature mismatch**  
The internal property head learns to predict from post-Transformer graph embeddings. The external oracle uses Morgan ECFP4 fingerprints. These feature spaces are structurally different. A head that accurately predicts from graph embeddings does not guarantee oracle alignment. The generator can satisfy the head without moving the oracle score.

**C. High-dimensional sparse REINFORCE**  
The generator's action space is 45 × 14 (atoms) + 45² × 5 (bonds) ≈ 10,755 discrete decisions per molecule. A single scalar reward per molecule provides extremely sparse gradient information. The effective signal-to-noise ratio at 500 steps is near zero, causing random walk behaviour (negative correlations worse than baseline).

**D. Out-of-distribution generation**  
The generator produces molecules with predicted pIC50 ≈ 5.4–5.5 regardless of condition. This reflects the generator's natural mode (drug-like molecules without BACE1-specific pharmacophores). No conditioning approach escaped this attractor basin within 500 steps.

---

## 7. Post-Hoc Oracle Filtering

Since conditioning failed, the project fell back to **large-scale generation + oracle filtering**.

**Generation run:**
- Model: `Issue3-ChemistryReadout` baseline, requested pIC50 = 7.0 (neutral), seed = 42
- Total generated: 16,384 graphs
- Valid decoded: 11,668 (71.2%)
- Unique canonical SMILES after deduplication: 11,376 (292 removed — all duplicate trivial molecules: 100 × "O" water, 70 × "C" methane, etc.)

**Oracle predictions (verified from `oracle_pool_16384_pred.csv`):**

| Threshold | Count | % of 11,376 |
|-----------|-------|-------------|
| pIC50 ≥ 6.0 | 318 | 2.8% |
| pIC50 ≥ 7.0 | 5 | 0.04% |
| pIC50 ≥ 7.6576 (original threshold) | 3 | 0.03% |

**Maximum oracle prediction:** 7.904 — predicted logBB = −0.020 (BBB-negative; does NOT meet joint threshold)

**ZERO molecules meet both original thresholds simultaneously.**

---

## 8. Oracle Model Validation

**Model:** ExtraTrees Regressor, Morgan ECFP4 (radius=2, nBits=2048)  
**Split:** Scaffold-separated (2,082/404/439 unique scaffolds in train/val/test; disjoint verified)

| Metric | Value | Note |
|--------|-------|------|
| Training-set Pearson r | ~0.901 | **IN-SAMPLE FIT ONLY — must not be reported as external accuracy** |
| Test R² (n=1,240) | **0.6156** | Correct external accuracy metric |
| Test RMSE | **0.7553 pIC50 units** | Typical prediction error |
| Test MAE | **0.5879 pIC50 units** | Median-like error |

**Uncertainty note:** RMSE = 0.755 is a typical (root-mean-square) error, not a calibrated 95% prediction interval. A formal calibrated interval was not computed. Generated molecules with novel scaffolds may fall outside the oracle's applicability domain.

**Coverage at threshold:** 44.4% of training molecules have pIC50 ≥ 7.0, providing reasonable training coverage near the joint threshold. However, no saved test-set predictions were found; Pearson r and Spearman rho on the held-out test cannot be computed from available files.

---

## 9. BBB Model Validation

**Model:** ExtraTrees Regressor, Morgan ECFP4 (radius=2, nBits=2048)  
**Training data:** B3DB regression dataset, 1,058 experimental logBB values  
**Split:** Scaffold-separated (742/177/139 train/val/test)

| Metric | Value |
|--------|-------|
| Test R² (n=139) | **0.354** |
| Test RMSE | **0.621 log units** |
| Test MAE | **0.484 log units** |

**Critical limitations:**
1. B3DB contains CNS drugs and probe compounds; BACE1 inhibitors are structurally distinct (larger, more lipophilic, peptidomimetic). The applicability domain for BACE1 scaffolds is **unverified**.
2. `predicted_logBB` in the BACE1 dataset is a model prediction, not an experimental measurement. BBB filtering for generated candidates is therefore prediction-of-prediction with cumulative uncertainty.
3. Given RMSE = 0.621 log units, a molecule predicted at logBB = +0.023 (the best clean candidate) plausibly has true logBB anywhere from approximately −0.6 to +0.6.

---

## 10. Docking Methodology and Results

**Software:** AutoDock Vina v1.2.7  
**Receptor:** BACE1 crystal structure PDB 2B8V, chain A (2,939 ATOM records)  
**Box:** Centred on co-crystallised ligand 3BN centroid (30.65, 41.54, 1.86 Å), 22 Å³ box  
**Candidates:** Top 50 Ro5-passing molecules by oracle pIC50

**Receptor preparation — known limitations:**
- Charges assigned via atom-type lookup table (not Gasteiger/MMFF computed)
- Catalytic dyad Asp32 and Asp228 present in chain A but **protonation state not explicitly set** — critical for BACE1 binding
- All crystallographic waters and cofactors removed
- Redocking validation of co-crystallised ligand 3BN: **failed at ligand preparation step** (meeko preparation error); quantitative RMSD validation could not be performed

**Results (verified from `bace1_final_ranked.csv`):**

| Metric | Value |
|--------|-------|
| Molecules submitted | 50 |
| Successful dockings | **50** (prior "37" figure was from stale intermediate summary) |
| Best Vina score | **−9.888 kcal/mol** |
| Mean Vina score | −7.810 kcal/mol |
| BBB+ (logBB ≥ 0) AND Vina ≤ −7.5 | **11** |

**Scores are for relative ranking within this study only and must not be compared to literature values from properly prepared receptors or converted to experimental binding affinities.**

---

## 11. Medicinal Chemistry Validation

Applied to the 11 dual-success (BBB+ AND Vina ≤ −7.5) candidates using RDKit FilterCatalog.

**Filters:** PAINS A/B/C, Brenk structural alerts, reactive group SMARTS

| Result | Count |
|--------|-------|
| CLEAN (no alerts, QED ≥ 0.5) | **1** |
| CLEAN (no alerts, QED < 0.5) | **1** |
| Brenk het-C-het_not_in_ring | 7 |
| Imine-reactive (aminothiazine class) | 2 |

**Scaffold analysis (top 200 by oracle pIC50):**
- 186/200 unique Murcko scaffolds (93%) — verified ✓
- Mean pairwise Tanimoto similarity: 0.146 — verified ✓
- Exact training-set duplicates (Tc ≥ 0.999): 3/200

**Note on imine alerts:** The 2 imine-flagged candidates have aminothiazine/iminopyrimidine cores — the same scaffold class as verubecestat (a clinical-stage BACE1 inhibitor). The Brenk imine alert may be over-conservative for this scaffold class. Individual structure review by a medicinal chemist is recommended before dismissing these candidates.

---

## 12. Outcome Categories and Key Candidate Profiles

### Outcome Categories

| Category | Description | Count |
|----------|-------------|-------|
| **A** | Molecules meeting BOTH original thresholds (pIC50 ≥ 7.6576 AND logBB ≥ 0) | **0** |
| **B** | BBB-positive AND Vina ≤ −7.5 kcal/mol | 11 |
| **C** | CLEAN (no structural alerts) among B | 2 |
| **D** | High predicted pIC50 (≥7.0), fail BBB criterion | 5 |

### Maximum Oracle pIC50 Candidate

**SMILES:** `CC(C)(C)COc1cnc2c(c1)C1(COC(N)=N1)c1cc(-c3cncnc3)ccc1O2`

| Property | Value | Threshold | Passes? |
|----------|-------|-----------|---------|
| Oracle pIC50 | 7.904 | ≥ 7.6576 | **YES** |
| Predicted logBB | −0.020 | ≥ 0.0 | **NO** |
| Vina score | −7.21 kcal/mol | — | — |
| Meets original joint threshold | — | — | **NO** |

Scaffold: iminopyrimidine core with tert-butyl ether and pyrimidine substituents. BBB-negative at the model threshold.

### Best Fully Clean Candidate

**SMILES:** `CC1CCC(COc2ccc(F)cn2)CN1C(O)c1cc(F)ccc1-c1ncccn1`

| Property | Value | Threshold | Passes? |
|----------|-------|-----------|---------|
| Oracle pIC50 | 6.169 | ≥ 7.6576 | **NO** |
| Predicted logBB | +0.023 | ≥ 0.0 | **YES** |
| Vina score | −8.03 kcal/mol | ≤ −7.5 | **YES** |
| QED | 0.643 | — | — |
| MW | 426.5 | ≤ 500 | **YES** |
| Stereocenters | 3 | — | — |
| PAINS | None | — | **CLEAN** |
| Brenk alerts | None | — | **CLEAN** |
| Meets original joint threshold | — | — | **NO** |

These are **two distinct molecules**. The maximum-activity candidate fails BBB; the best clean candidate fails pIC50.

---

## 13. Limitations and Uncertainty

1. **Primary objective not met:** Zero molecules computationally satisfy both original thresholds
2. **Oracle accuracy:** Scaffold-split test R² = 0.616, RMSE = 0.755 pIC50 units — predictions carry ~0.75 unit typical error; no formal calibrated prediction interval
3. **BBB model accuracy:** Scaffold-split test R² = 0.354, RMSE = 0.621 log units; BACE1 scaffolds are outside B3DB training domain — applicability unverified
4. **Conditioning failure:** All 8 approaches failed to produce condition-responsive generation without validity collapse
5. **Docking receptor:** Simplified PDBQT without proper protonation/charges; scores are for relative ranking only; redocking RMSD validation could not be performed
6. **Trivial molecules:** 292/11,668 valid-flagged molecules were degenerate (water, methane, etc.) — removed before oracle evaluation
7. **No experimental validation:** All findings are computational predictions requiring synthesis and wet-lab assay

---

## 14. Prioritised Next Steps

### Immediate (no new experiments)
1. **BBB applicability domain:** Compute Tanimoto similarity from BACE1 candidates to B3DB training set; flag molecules outside applicability domain
2. **Receptor preparation:** Proper protonation of D32/D228 using H++ or HTMD before further docking interpretations

### Short-term computational
3. **Cross-attention conditioning pilot:** Implement condition injection via cross-attention (condition as K/V, node representations as Q) — tested in Phase 6 of this report
4. **Larger generation pool:** Generate 100k+ molecules for filtering; expected to yield ~30 candidates with pIC50 ≥ 7.0

### For experimental follow-up
5. **Synthesise 2 clean candidates:** Even moderate activity (pIC50 > 5.5) in vitro would validate the computational pipeline
6. **PAMPA BBB assay:** Confirm or refute the predicted logBB ≥ 0 for the 11 dual candidates

### For publication
7. **Baseline comparison:** Add REINVENT4 or MOLDIFF as a comparative method
8. **Full ADMET profiling:** pkCSM or SwissADME for hERG, CYP, metabolic stability
9. **Report oracle accurately:** Use test R² = 0.616 and RMSE = 0.755; never cite training-set r = 0.901 as external accuracy

---

## Phase 6: Cross-Attention Conditioning Pilot

**Decision:** YES — cross-attention was warranted as the one untested architectural change that provides position-selective condition injection.

**Hypothesis:** Cross-attention (condition as K/V, node representations as Q) would provide each atom position with a selective signal rather than the uniform additive bias causing saturation.

**Implementation:** Added `nn.MultiheadAttention` after `TransformerEncoder`, before `readout_condition`. Residual connection. 46 existing parameters transferred, 6 new initialized. `strict=False` warm-start from `Issue3-ChemistryReadout`.

### Training outcome (500 steps, λ_condition=5)

| Metric | Baseline | Cross-Attention |
|--------|----------|-----------------|
| corr(req, head) | 0.364 | **0.477** (+31%) |
| G→head cond MSE | 1.616 | **1.054** (−35%) |

The cross-attention improved internal proxy metrics — a real signal.

### Generation outcome

| Condition | Valid / 1024 | ValidFrac | Δ pIC50 |
|-----------|-------------|-----------|---------|
| pIC50=5.5 | 578 | 56.4% ✓ | — |
| **pIC50=8.0** | **0** | **0.0%** ✗ | **N/A** |

**Result: FAIL.** The pIC50=8.0 condition produced zero valid molecules — the same saturation collapse as the λ_condition=5 sweep without cross-attention.

**Key insight:** Saturation is driven by the **condition loss gradient at λ=5**, not by the injection mechanism (additive vs. cross-attention). The internal head pressure to push atoms away from the padding class overwhelms validity regardless of how the condition signal is delivered. This is a fundamental property of the current training objective at high λ, not an architectural limitation of additive embedding alone.

**Conclusion:** Cross-attention alone is insufficient. 9 consecutive conditioning approaches have failed.

**Recommended next step:** Scale the post-hoc oracle filtering pipeline to 500k molecules (expected ~140 candidates with pIC50 ≥ 7.0 based on observed 2.8% yield at pIC50 ≥ 6.0). Further conditioning improvements require a fundamentally different training paradigm.

*Full decision report: `reports/cross_attention_decision.md`*  
*Results: `results/cross_attention_pilot/pilot_results.json`*

---

## Phase 7: Staged Post-Hoc Oracle Screening (50k Pilot)

### Motivation
Nine conditioning approaches failed to push the generator's output above the target thresholds.
This phase scales post-hoc filtering instead: run the baseline model with a neutral condition
(pIC50=7.0, logBB=0.0), generate a large pool, and screen by the independent oracle.

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | Issue3-ChemistryReadout baseline (unchanged) |
| Condition | Requested pIC50=7.0, logBB=0.0 (neutral, same as 16k run) |
| Graphs targeted | 50,048 (batch_size=128, num_batches=391) |
| Seed | 42 (matches 16k run for comparability; not an exact replay) |
| pIC50 threshold | >= 7.6576 (unchanged) |
| logBB threshold | >= 0.0 (unchanged) |

### Key Results

| Metric | Value |
|--------|-------|
| Total graphs generated | 50,048 |
| Valid molecules | 35,874 (validity fraction = 71.68%) |
| Unique evaluated molecules | 35,042 |
| Mean predicted pIC50 | 5.479 |
| Max predicted pIC50 | 7.921 |
| pIC50 >= 6.0 | 1,073 (3.06%) |
| pIC50 >= 7.0 | 17 (0.049%) |
| pIC50 >= 7.6576 | 6 (0.017%) |
| logBB >= 0.0 | 22,282 (63.59%) |
| **BOTH thresholds** | **0** |

The validity fraction (71.68%) is stable and consistent with the 16k run. No atom-class
collapse was observed. The mean predicted pIC50 (5.479) confirms the generator's natural
mode is unchanged regardless of the requested condition.

### Comparison to 16k Run

| Metric | 16k Run | 50k Pilot | Change |
|--------|---------|-----------|--------|
| Graphs generated | 16,384 | 50,048 | +33,664 |
| Valid molecules | 11,668 | 35,874 | +24,206 |
| Validity fraction | 0.7122 | 0.7168 | +0.005 |
| Unique evaluated | 11,376 | 35,042 | +23,666 |
| pIC50 >= 6.0 | 318 | 1,073 | +755 |
| pIC50 >= 7.0 | 5 | 17 | +12 |
| pIC50 >= 7.6576 | 3 | 6 | +3 |
| Max pIC50 | 7.9037 | 7.9210 | +0.017 |
| logBB >= 0.0 | 7,701 | 22,282 | +14,581 |
| BOTH thresholds | 0 | **0** | 0 |
| Rate pIC50 >= 6.0 | 2.80% | 3.06% | +0.27 pp |

Both runs used seed 42. The 50k pilot used a different number of batches (391 vs 128 in the
16k run), so the molecule pools overlap but are not identical. The joint-pass count is
zero in both runs — a consistent, reproducible finding across 46,418 total unique evaluated
molecules.

### Joint-Threshold Hits

**Zero molecules met both original thresholds (pIC50 >= 7.6576 AND logBB >= 0.0).**

The 6 molecules with predicted pIC50 >= 7.6576 all have predicted logBB < 0.0. The
maximum oracle prediction is 7.921 (logBB = −0.014). The negative correlation between
high predicted pIC50 and logBB >= 0 reflects the generator's structural output: molecules
carrying the highest-activity BACE1 pharmacophores (iminopyrimidine / spiro-bicyclic
cores) are systematically predicted as BBB-negative by the ExtraTrees logBB model.

### Scale-Up Decision

All four pre-specified criteria were met:

| Criterion | Required | Actual | Met? |
|-----------|----------|--------|------|
| Validity fraction >= 0.50 | ≥ 50% | 71.68% | YES |
| Unique evaluated >= 25,000 | ≥ 25,000 | 35,042 | YES |
| Rate pIC50 >= 6.0 > 0 | > 0% | 3.06% | YES |
| No atom-class collapse | None | None | YES |

**Scale-up to 500,000 graphs is recommended.**

Scale-up command (seed 43, num_batches 3907, ~500,096 target graphs):

```bash
.venv\Scripts\python.exe generate_joint_molecules.py \
  --checkpoint experiments/models/Issue3-ChemistryReadout/joint_generator.pt \
  --batch_size 128 --num_batches 3907 \
  --desired_pIC50 7.0 --desired_logBB 0.0 \
  --seed 43 \
  --output results/baseline_screen_500000_generated.csv
```

Estimated runtime: ~4.5 hours. Estimated disk: ~700 MB total (generated + predictions).

Illustrative projection at 500k scale (~350,084 unique evaluated molecules at observed rates):
~10,720 molecules with pIC50 >= 6.0; ~170 with pIC50 >= 7.0.

> **Statistical extrapolation only. Not a guarantee. Joint-pass yield cannot be reliably
> inferred from marginal rates alone.**

*Full pilot report: `results/baseline_screen_50000_report.md`*  
*Summary statistics: `results/baseline_screen_50000_summary.json`*  
*Comparison table: `results/baseline_screen_50000_comparison.csv`*

---

## Verified File Paths

| Artifact | Path |
|---------|------|
| Master candidate table (11,376 rows) | `results/bace1_master_candidate_table.csv` |
| Exclusions table (292 rows) | `results/bace1_candidate_exclusions.csv` |
| Final ranked docking table (50 rows) | `results/docking/bace1_final_ranked.csv` |
| Validated candidates (11 rows) | `results/docking/bace1_validated_candidates.csv` |
| Oracle model metadata | `results/bace1_activity_model/model_metadata.json` |
| BBB model metadata | `results/bbb_activity_model/bbb_model_metadata.json` |
| Docking reconciliation | `reports/audit_corrections/docking_reconciliation.csv` |
| Oracle exclusion reconciliation | `reports/audit_corrections/oracle_exclusion_reconciliation.csv` |
| Oracle metrics (verified) | `reports/audit_corrections/oracle_metrics_verified.json` |
| Novelty analysis top-200 | `reports/audit_corrections/novelty_analysis_top200.csv` |
| BBB model audit | `reports/audit_corrections/bbb_model_audit.txt` |
| Oracle training coverage | `reports/audit_corrections/oracle_training_coverage.txt` |
| Receptor audit | `reports/audit_corrections/receptor_audit.txt` |
| Docking limitations | `reports/audit_corrections/docking_limitations.txt` |
| Conditioning experiment logs | `lambda_sweep_results.txt`, `vgated_sweep2_results.txt`, `reinforce_pilot_v4_log.txt` |

---

*This report was compiled from verified file contents. All numerical claims are sourced from actual saved artifacts. Unresolved gaps are identified explicitly.*

---

## Phase 8: Candidate-Level Validation of High-Activity Molecules

### Validation Methodology

A rigorous candidate-level validation was conducted on all 6 molecules from the
high-activity candidate audit, with detailed profiles for focus candidates #1, #3, #5, #6.
Validation included: RDKit-computed physicochemical descriptors; PAINS, Brenk, and NIH
reactive-group filters; Morgan ECFP4 (radius=2) training-set Tanimoto similarity;
B3DB structural similarity for BBB applicability domain assessment; and cross-run
deduplication against the 16k initial pool.

### Focus Candidate Summary

| # | pIC50 | logBB | QED | Brenk | Max Train Tc | Novelty Class | Priority |
|---|-------|-------|-----|-------|-------------|---------------|----------|
| #1 | 7.9037 | -0.0202 | 0.693 | none | 1.000 | NEAR_DUP | 3 |
| #3 | 7.7494 | -0.0324 | 0.865 | none | 0.846 | NOVEL | 1 |
| #5 | 7.7400 | -0.2242 | 0.777 | none | 0.701 | NOVEL | 2 |
| #6 | 7.9210 | -0.0142 | 0.618 | 2-halo_pyridine | 1.000 | NEAR_DUP | 4 |

### BBB Prediction Uncertainty

All 6 candidates have predicted logBB < 0.0 — **none meets the logBB >= 0.0 threshold**.
The BBB regressor (ExtraTrees, R²=0.354, RMSE=0.621) was trained on B3DB
(CNS drugs). BACE1 peptidomimetic scaffolds are out-of-domain for this training set.

Candidates #1 (logBB=−0.020), #3 (logBB=−0.032), and #6 (logBB=−0.014) are each
within 0.06 logBB units of the 0.0 threshold — less than 0.10 RMSE-units — and must
be classified as **INDETERMINATE**. The prediction cannot reliably confirm or deny BBB
penetrance for these molecules. Candidate #5 (logBB=−0.224) is **LIKELY_NEGATIVE**.

### Oracle Reliability and Applicability Domain

BACE1 oracle (ExtraTrees, R²=0.616, RMSE=0.755; scaffold-split test).

- **#5** (Tc=0.702): lowest interpolation support; prediction confidence is reduced.
- **#3** (Tc=0.846): better interpolation support; most credible novel prediction.
- **#1 and #6** (Tc=1.000): exact isomeric duplicates of training compounds.
  Their predicted pIC50 values reflect training-set memorisation, not generalisation.
  Training-set high-activity coverage (pIC50>=7.6576): 2051/8271 (24.8%).
  Sparse coverage in the high-pIC50 region increases oracle uncertainty for all candidates.

### Novelty Finding

- **#1** (Tc=1.000): exact isomeric duplicate of training compound. NOT novel.
- **#6** (Tc=1.000): exact isomeric duplicate of training compound. NOT novel.
- **#3** (Tc=0.846): not an exact duplicate; structurally analogous but **NOVEL**.
- **#5** (Tc=0.702): **NOVEL**; structurally more distinct from the training set.

### Answers to Seven Research Questions

**Q1 — Strongest candidate:** Candidate #3 (pIC50=7.749, logBB=−0.032, QED=0.865, no alerts, novel).

**Q2 — Novelty:** #3 and #5 are novel. #1 and #6 are exact isomeric training-set duplicates.

**Q3 — BBB reliability near zero:** No. Model R²=0.354, RMSE=0.621, out-of-domain scaffolds.
BBB status is INDETERMINATE for #1, #3, #6 and LIKELY_NEGATIVE for #5.

**Q4 — Oracle support:** Sufficient for #3 (Tc=0.846); weaker for #5 (Tc=0.702).
#1 and #6 are training duplicates; their predictions are not informative.

**Q5 — Both thresholds satisfied:** NO. Every candidate has logBB < 0. Zero molecules
satisfy both pIC50 >= 7.6576 AND logBB >= 0.0 across all generation runs.

**Q6 — Scale-up vs. validate:** Validate existing candidates first. The BBB model's
domain mismatch is the limiting factor; a larger run will not resolve this.

**Q7 — Responsible claim to professor:** DrugGEN generated structurally diverse molecules
with competitive predicted BACE1 activity. The primary research finding is that zero
molecules met the dual-property criterion. Candidate #3 is the strongest computationally
prioritised novel candidate and warrants experimental follow-up for both BACE1 activity
and BBB penetrance validation.

### Final Recommendation

**Candidate #3** is the strongest novel, clean, computationally-prioritised candidate.
However, **no candidate satisfies both original thresholds** (pIC50 >= 7.6576 AND
logBB >= 0.0). This is an honest and reportable finding. Experimental validation
(synthesis, in vitro BACE1 IC50, PAMPA or MDCK-MDR1 BBB assay) of candidate #3
is recommended as the next step.

*Phase 8 validation completed. All results are computational estimates.*

---

## Phase 8: Candidate-Level Validation of High-Activity Molecules

### Validation Methodology

A rigorous candidate-level validation was conducted on all 6 molecules from the
high-activity candidate audit, with detailed profiles for focus candidates #1, #3, #5, #6.
Validation included: RDKit-computed physicochemical descriptors; PAINS, Brenk, and NIH
reactive-group filters; Morgan ECFP4 (radius=2) training-set Tanimoto similarity;
B3DB structural similarity for BBB applicability domain assessment; and cross-run
deduplication against the 16k initial pool.

### Focus Candidate Summary

| # | pIC50 | logBB | QED | Brenk | Max Train Tc | Novelty Class | Priority |
|---|-------|-------|-----|-------|-------------|---------------|----------|
| #1 | 7.9037 | -0.0202 | 0.693 | none | 1.000 | NEAR_DUP | 3 |
| #3 | 7.7494 | -0.0324 | 0.865 | none | 0.846 | NOVEL | 1 |
| #5 | 7.7400 | -0.2242 | 0.777 | none | 0.701 | NOVEL | 2 |
| #6 | 7.9210 | -0.0142 | 0.618 | 2-halo_pyridine | 1.000 | NEAR_DUP | 4 |

### BBB Prediction Uncertainty

All 6 candidates have predicted logBB < 0.0 — **none meets the logBB >= 0.0 threshold**.
The BBB regressor (ExtraTrees, R²=0.354, RMSE=0.621) was trained on B3DB
(CNS drugs). BACE1 peptidomimetic scaffolds are out-of-domain for this training set.

Candidates #1 (logBB=−0.020), #3 (logBB=−0.032), and #6 (logBB=−0.014) are each
within 0.06 logBB units of the 0.0 threshold — less than 0.10 RMSE-units — and must
be classified as **INDETERMINATE**. The prediction cannot reliably confirm or deny BBB
penetrance for these molecules. Candidate #5 (logBB=−0.224) is **LIKELY_NEGATIVE**.

### Oracle Reliability and Applicability Domain

BACE1 oracle (ExtraTrees, R²=0.616, RMSE=0.755; scaffold-split test).

- **#5** (Tc=0.702): lowest interpolation support; prediction confidence is reduced.
- **#3** (Tc=0.846): better interpolation support; most credible novel prediction.
- **#1 and #6** (Tc=1.000): exact isomeric duplicates of training compounds.
  Their predicted pIC50 values reflect training-set memorisation, not generalisation.
  Training-set high-activity coverage (pIC50>=7.6576): 2051/8271 (24.8%).
  Sparse coverage in the high-pIC50 region increases oracle uncertainty for all candidates.

### Novelty Finding

- **#1** (Tc=1.000): exact isomeric duplicate of training compound. NOT novel.
- **#6** (Tc=1.000): exact isomeric duplicate of training compound. NOT novel.
- **#3** (Tc=0.846): not an exact duplicate; structurally analogous but **NOVEL**.
- **#5** (Tc=0.702): **NOVEL**; structurally more distinct from the training set.

### Answers to Seven Research Questions

**Q1 — Strongest candidate:** Candidate #3 (pIC50=7.749, logBB=−0.032, QED=0.865, no alerts, novel).

**Q2 — Novelty:** #3 and #5 are novel. #1 and #6 are exact isomeric training-set duplicates.

**Q3 — BBB reliability near zero:** No. Model R²=0.354, RMSE=0.621, out-of-domain scaffolds.
BBB status is INDETERMINATE for #1, #3, #6 and LIKELY_NEGATIVE for #5.

**Q4 — Oracle support:** Sufficient for #3 (Tc=0.846); weaker for #5 (Tc=0.702).
#1 and #6 are training duplicates; their predictions are not informative.

**Q5 — Both thresholds satisfied:** NO. Every candidate has logBB < 0. Zero molecules
satisfy both pIC50 >= 7.6576 AND logBB >= 0.0 across all generation runs.

**Q6 — Scale-up vs. validate:** Validate existing candidates first. The BBB model's
domain mismatch is the limiting factor; a larger run will not resolve this.

**Q7 — Responsible claim to professor:** DrugGEN generated structurally diverse molecules
with competitive predicted BACE1 activity. The primary research finding is that zero
molecules met the dual-property criterion. Candidate #3 is the strongest computationally
prioritised novel candidate and warrants experimental follow-up for both BACE1 activity
and BBB penetrance validation.

### Final Recommendation

**Candidate #3** is the strongest novel, clean, computationally-prioritised candidate.
However, **no candidate satisfies both original thresholds** (pIC50 >= 7.6576 AND
logBB >= 0.0). This is an honest and reportable finding. Experimental validation
(synthesis, in vitro BACE1 IC50, PAMPA or MDCK-MDR1 BBB assay) of candidate #3
is recommended as the next step.

*Phase 8 validation completed. All results are computational estimates.*

---

## Phase 8: Candidate-Level Validation of High-Activity Molecules

### Validation Methodology

A rigorous candidate-level validation was conducted on all 6 molecules from the
high-activity candidate audit, with detailed profiles for focus candidates #1, #3, #5, #6.
Validation included: RDKit-computed physicochemical descriptors; PAINS, Brenk, and NIH
reactive-group filters; Morgan ECFP4 (radius=2) training-set Tanimoto similarity;
B3DB structural similarity for BBB applicability domain assessment; and cross-run
deduplication against the 16k initial pool.

### Focus Candidate Summary

| # | pIC50 | logBB | QED | Brenk | Max Train Tc | Novelty Class | Priority |
|---|-------|-------|-----|-------|-------------|---------------|----------|
| #1 | 7.9037 | -0.0202 | 0.693 | none | 1.000 | NEAR_DUP | 3 |
| #3 | 7.7494 | -0.0324 | 0.865 | none | 0.846 | NOVEL | 1 |
| #5 | 7.7400 | -0.2242 | 0.777 | none | 0.701 | NOVEL | 2 |
| #6 | 7.9210 | -0.0142 | 0.618 | 2-halo_pyridine | 1.000 | NEAR_DUP | 4 |

### BBB Prediction Uncertainty

All 6 candidates have predicted logBB < 0.0 — **none meets the logBB >= 0.0 threshold**.
The BBB regressor (ExtraTrees, R²=0.354, RMSE=0.621) was trained on B3DB
(CNS drugs). BACE1 peptidomimetic scaffolds are out-of-domain for this training set.

Candidates #1 (logBB=−0.020), #3 (logBB=−0.032), and #6 (logBB=−0.014) are each
within 0.06 logBB units of the 0.0 threshold — less than 0.10 RMSE-units — and must
be classified as **INDETERMINATE**. The prediction cannot reliably confirm or deny BBB
penetrance for these molecules. Candidate #5 (logBB=−0.224) is **LIKELY_NEGATIVE**.

### Oracle Reliability and Applicability Domain

BACE1 oracle (ExtraTrees, R²=0.616, RMSE=0.755; scaffold-split test).

- **#5** (Tc=0.702): lowest interpolation support; prediction confidence is reduced.
- **#3** (Tc=0.846): better interpolation support; most credible novel prediction.
- **#1 and #6** (Tc=1.000): exact isomeric duplicates of training compounds.
  Their predicted pIC50 values reflect training-set memorisation, not generalisation.
  Training-set high-activity coverage (pIC50>=7.6576): 2051/8271 (24.8%).
  Sparse coverage in the high-pIC50 region increases oracle uncertainty for all candidates.

### Novelty Finding

- **#1** (Tc=1.000): exact isomeric duplicate of training compound. NOT novel.
- **#6** (Tc=1.000): exact isomeric duplicate of training compound. NOT novel.
- **#3** (Tc=0.846): not an exact duplicate; structurally analogous but **NOVEL**.
- **#5** (Tc=0.702): **NOVEL**; structurally more distinct from the training set.

### Answers to Seven Research Questions

**Q1 — Strongest candidate:** Candidate #3 (pIC50=7.749, logBB=−0.032, QED=0.865, no alerts, novel).

**Q2 — Novelty:** #3 and #5 are novel. #1 and #6 are exact isomeric training-set duplicates.

**Q3 — BBB reliability near zero:** No. Model R²=0.354, RMSE=0.621, out-of-domain scaffolds.
BBB status is INDETERMINATE for #1, #3, #6 and LIKELY_NEGATIVE for #5.

**Q4 — Oracle support:** Sufficient for #3 (Tc=0.846); weaker for #5 (Tc=0.702).
#1 and #6 are training duplicates; their predictions are not informative.

**Q5 — Both thresholds satisfied:** NO. Every candidate has logBB < 0. Zero molecules
satisfy both pIC50 >= 7.6576 AND logBB >= 0.0 across all generation runs.

**Q6 — Scale-up vs. validate:** Validate existing candidates first. The BBB model's
domain mismatch is the limiting factor; a larger run will not resolve this.

**Q7 — Responsible claim to professor:** DrugGEN generated structurally diverse molecules
with competitive predicted BACE1 activity. The primary research finding is that zero
molecules met the dual-property criterion. Candidate #3 is the strongest computationally
prioritised novel candidate and warrants experimental follow-up for both BACE1 activity
and BBB penetrance validation.

### Final Recommendation

**Candidate #3** is the strongest novel, clean, computationally-prioritised candidate.
However, **no candidate satisfies both original thresholds** (pIC50 >= 7.6576 AND
logBB >= 0.0). This is an honest and reportable finding. Experimental validation
(synthesis, in vitro BACE1 IC50, PAMPA or MDCK-MDR1 BBB assay) of candidate #3
is recommended as the next step.

*Phase 8 validation completed. All results are computational estimates.*
