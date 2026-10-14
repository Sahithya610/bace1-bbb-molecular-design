# Baseline Screen 50k Pilot — Experiment Report

**Date:** October 2026  
**Status:** Computational only — no experimental validation  
**Author:** Computational Drug Discovery Study

> **Terminology note.** "Predicted pIC50" refers to ExtraTrees oracle model output. "Predicted logBB"
> refers to a separate B3DB-trained ExtraTrees regression model output. Neither constitutes an
> experimentally measured value. No synthesis or in-vitro validation has been performed.

---

## 1. Experiment Configuration and Provenance

| Parameter | Value |
|-----------|-------|
| Checkpoint | `experiments/models/Issue3-ChemistryReadout/joint_generator.pt` |
| Requested pIC50 (condition) | 7.0 |
| Requested logBB (condition) | 0.0 |
| Batch size | 128 |
| Number of batches | 391 |
| Total target graphs | 50,048 |
| Random seed | 42 |
| BACE1 oracle | ExtraTrees Regressor, Morgan ECFP4 (radius=2, nBits=2048) |
| Oracle test R² | 0.6156 (scaffold-held-out split, n=1,240) |
| Oracle test RMSE | 0.755 pIC50 units |
| Oracle test MAE | 0.588 pIC50 units |
| BBB model | ExtraTrees Regressor, Morgan ECFP4 (radius=2, nBits=2048) |
| BBB model test R² | 0.354 (scaffold-held-out split, n=139) |
| BBB model note | Limited; BACE1 scaffolds are out-of-domain for B3DB training set |
| pIC50 threshold | >= 7.6576 |
| logBB threshold | >= 0.0 |
| Joint criterion | BOTH thresholds must be met by the same unique valid molecule |

**Seed note:** Seed 42 matches the previous 16,384-graph run for comparability. This is NOT an
exact replay — the 16k run used a different number of batches (128). The generation order and
molecule pool will differ accordingly.

**Condition normalization (verified):** pIC50 normalized via (7.0 − 6.691) / 1.309 = +0.236;
logBB normalized via (0.0 − (−0.078)) / 0.752 = +0.104. Both fall well within the
generator's in-distribution condition range.

---

## 2. Generation Statistics

All figures read from `results/baseline_screen_50000_summary.json`.

| Metric | Value |
|--------|-------|
| Total graphs generated | 50,048 |
| Valid molecules (decoded) | 35,874 |
| Validity fraction | 0.7168 (71.68%) |
| Duplicate records removed | 832 |
| Unique evaluated molecules | 35,042 |
| Uniqueness fraction of valid | 0.9768 (97.68%) |
| Invalid / undecoded records | 14,174 |

The generator produced valid molecules at a stable rate consistent with the 16k run (71.7% vs
71.2%). No atom-class collapse was observed. The 832 duplicates are a modest fraction (2.3% of
valid molecules) consistent with prior runs.

---

## 3. Oracle Statistics

All figures read from `results/baseline_screen_50000_summary.json`.

### Predicted pIC50

| Statistic | Value |
|-----------|-------|
| Mean | 5.4785 |
| Median | 5.4692 |
| Standard deviation | 0.2735 |
| Minimum | 4.3293 |
| Maximum | 7.9210 |

### Predicted logBB

| Statistic | Value |
|-----------|-------|
| Mean | 0.0746 |
| Median | 0.0730 |
| Standard deviation | 0.2042 |
| Minimum | −1.0527 |
| Maximum | 1.0343 |

The mean predicted pIC50 (5.48) is close to the distribution observed in the 16k run (~5.47),
confirming that the generator's natural mode has not shifted. The generator does not strongly
respond to the requested pIC50 = 7.0 condition, consistent with the finding from all 9
conditioning experiments that conditioning has limited effect on oracle-predicted activity.

---

## 4. Threshold Counts and Rates

All figures read from `results/baseline_screen_50000_summary.json`, `threshold_counts` field.
Denominator for all rates: 35,042 unique evaluated molecules.

| Threshold | Count | Rate |
|-----------|-------|------|
| pIC50 >= 6.0 | 1,073 | 3.062% |
| pIC50 >= 7.0 | 17 | 0.0485% |
| pIC50 >= 7.6576 (original threshold) | 6 | 0.0171% |
| logBB >= 0.0 | 22,282 | 63.587% |
| **Both original thresholds** | **0** | **0.0%** |

**Zero molecules met both original thresholds simultaneously in this pilot.**

---

## 5. Top-10 Candidates by Predicted pIC50

All data read from `results/baseline_screen_50000_summary.json`, `top_10_by_pic50` field.

> **Important:** The values below are oracle model predictions, not experimental measurements.
> The BACE1 ExtraTrees oracle has test RMSE = 0.755 pIC50 units on a scaffold-held-out split.
> These values should not be interpreted as guaranteed experimental activity.

| Rank | Canonical SMILES | Predicted pIC50 | Predicted logBB | Passes pIC50 ≥ 7.6576? | Passes logBB ≥ 0.0? | Joint? |
|------|-----------------|-----------------|-----------------|------------------------|---------------------|--------|
| 1 | `NC1=NC2(CO1)c1cc(-c3cccnc3F)ccc1Oc1cnc(C3CCCOC3)cc12` | 7.9210 | −0.0142 | YES | NO | NO |
| 2 | `CC(C)(C)COc1cnc2c(c1)C1(COC(N)=N1)c1cc(-c3cncnc3)ccc1O2` | 7.9037 | −0.0202 | YES | NO | NO |
| 3 | `CCC1CC2(CCC1OC)Cc1ccc(-c3cncnc3)cc1C21COC(N)=N1` | 7.7494 | −0.0324 | YES | NO | NO |
| 4 | `CCCc1cc(-c2ccc3c(c2)C2(N=C(C)C(N)=N2)C2(CCC(OC)CC2)C3)ccn1` | 7.7400 | −0.2242 | YES | NO | NO |
| 5 | `COc1cnc(C(O)Nc2ccc3c(c2)C2(CCSC(N)=N2)c2cc(-c4ccnc(F)c4)nc(F)c2O3)cn1` | 7.7363 | −0.3161 | YES | NO | NO |
| 6 | `CCC(O)NC(Cc1cccc(-c2nccs2)c1)C(O)CNC1CC2(CCC2)Oc2ncc(CC(C)(C)C)cc21` | 7.7129 | −0.4340 | YES | NO | NO |
| 7 | `CN1OC2(N=C1N)c1cc(CCC3CC3)ccc1CC21CCC(OC(F)F)CC1` | 7.5529 | −0.0218 | NO | NO | NO |
| 8 | `COC1CCC2(CC1)Cc1ccc(CCC3CC3)cc1C21N=C(N)c2ccccc21` | 7.5369 | +0.1095 | NO | YES | NO |
| 9 | `NC1=NC2(c3ccc(F)cc3F)COC(c3ccc(F)c(F)c3)CC2CS1` | 7.4024 | +0.0353 | NO | YES | NO |
| 10 | `COC1CCC2(CC1)Cc1ccc(-c3cccc(CN)c3)cc1C21N=C(N)c2ccccc21` | 7.3916 | +0.0733 | NO | YES | NO |

The 6 molecules with predicted pIC50 ≥ 7.6576 all have predicted logBB < 0.0. The highest
logBB among top-10 candidates is +0.1095 (rank 8), but that molecule's predicted pIC50 is 7.537
— below the 7.6576 threshold. The maximum oracle score in this pilot is 7.921 with logBB = −0.014.

---

## 6. Joint-Threshold Hits

**Zero molecules met both original thresholds in this pilot.**

Applying both criteria simultaneously (predicted pIC50 >= 7.6576 AND predicted logBB >= 0.0) to
all 35,042 unique evaluated molecules returns an empty set. This is consistent with the 16k run,
which also returned zero joint hits.

The pattern is structural: molecules with the highest predicted pIC50 carry iminopyrimidine /
aminothiazine / spiro-bicyclic cores with hydrogen-bond donors. These scaffolds are associated
with high predicted BACE1 activity but are also relatively polar and lipophilic in a direction
that reduces predicted logBB. Molecule rank 8 above (logBB = +0.110) illustrates that the
generator can produce BBB-positive scaffolds, but not in combination with the high-activity
pharmacophore at the current generation scale.

---

## 7. Comparison Table with 16k Run

All figures read from `results/baseline_screen_50000_comparison.csv`.

| Metric | 16k Run | 50k Pilot | Change |
|--------|---------|-----------|--------|
| Graphs generated | 16,384 | 50,048 | +33,664 |
| Valid molecules | 11,668 | 35,874 | +24,206 |
| Validity fraction | 0.7122 | 0.7168 | +0.0046 |
| Unique evaluated | 11,376 | 35,042 | +23,666 |
| pIC50 >= 6.0 | 318 | 1,073 | +755 |
| pIC50 >= 7.0 | 5 | 17 | +12 |
| pIC50 >= 7.6576 | 3 | 6 | +3 |
| Max pIC50 | 7.9037 | 7.9210 | +0.017 |
| logBB >= 0.0 | 7,701 | 22,282 | +14,581 |
| BOTH thresholds | 0 | 0 | 0 |
| Rate pIC50 >= 6.0 | 2.7954% | 3.0620% | +0.267 pp |

**Differences between runs:**
- Seed 42 was used in both runs. Because the number of batches differs (the 16k run used 128
  batches; this pilot used 391), the molecule pools overlap but are not identical.
- The validity fraction is stable (within 0.5 percentage points), confirming no model or pipeline
  regression between runs.
- The rate of pIC50 >= 6.0 is slightly higher in the pilot (3.06% vs 2.80%), likely reflecting
  random variation rather than a systematic trend.
- The joint-pass count remains zero in both runs. This is a consistent, reproducible finding.

---

## 8. Scale-Up Decision

All criteria read from `results/baseline_screen_50000_summary.json`, `scaleup_decision` field.

| Criterion | Required | Actual | Met? |
|-----------|----------|--------|------|
| Validity fraction >= 0.50 | ≥ 50% | 71.68% | YES |
| Unique evaluated molecules >= 25,000 | ≥ 25,000 | 35,042 | YES |
| Rate pIC50 >= 6.0 > 0 | > 0% | 3.06% | YES |
| No atom-class collapse | None detected | None detected | YES |

**Recommendation: Scale-up to 500,000 graphs is recommended.**

All four pre-specified criteria are met. The generation pipeline is stable. The validity fraction
and pIC50 rate are reproducible across two independent runs. There is no evidence of graph
collapse or atom-class degeneration.

---

## 9. Scale-Up Command

Command read from `results/baseline_screen_50000_summary.json`, `scaleup_command` field.

```bash
.venv\Scripts\python.exe generate_joint_molecules.py \
  --checkpoint experiments/models/Issue3-ChemistryReadout/joint_generator.pt \
  --batch_size 128 \
  --num_batches 3907 \
  --desired_pIC50 7.0 \
  --desired_logBB 0.0 \
  --seed 43 \
  --output results/baseline_screen_500000_generated.csv
```

| Parameter | Value | Note |
|-----------|-------|------|
| Seed | 43 | Different from pilot (42); not a continuation of the same random sequence |
| num_batches | 3,907 | Targets 500,096 total graphs |
| Total target graphs | 500,096 | At 71.7% validity ≈ 358,500 valid; ≈ 350,000 unique evaluated |
| Estimated runtime | ~4.5 hours | On the same hardware used for this pilot |
| Estimated disk (generated CSV) | ~500 MB | |
| Estimated disk (predictions CSV) | ~200 MB | |

---

## 10. Illustrative Scale-Up Projection

All figures read from `results/baseline_screen_50000_summary.json`,
`scaleup_projection_illustrative` field.

Assuming the pilot's screening rates persist at 500k-graph scale
(estimated ~350,084 unique evaluated molecules at 71.7% validity × 97.7% uniqueness):

| Threshold | Pilot Rate | Projected Count (at 350,084 unique) |
|-----------|-----------|--------------------------------------|
| pIC50 >= 6.0 | 3.062% | ~10,720 |
| pIC50 >= 7.0 | 0.0485% | ~170 |

> **Statistical extrapolation only. Not a guarantee. Joint-pass yield cannot be reliably
> inferred from marginal rates alone.**

The joint-pass rate in both completed runs is 0/11,376 (16k) and 0/35,042 (50k). The two
marginal rates (pIC50 and logBB) are NOT independent — the highest-activity scaffolds in
this generator's output are systematically BBB-negative. A projection based on the product
of marginal rates would be misleading. The scale-up is justified to obtain more high-pIC50
candidates for structural analysis and possible medicinal-chemistry modification, not to
guarantee joint-threshold hits.

---

## 11. Limitations

1. **BACE1 oracle accuracy:** Test R² = 0.6156, RMSE = 0.755 pIC50 units on a scaffold-held-out
   split. All pIC50 values are model predictions. Typical prediction error is ~0.75 units; a
   molecule predicted at 7.92 plausibly has true pIC50 anywhere from roughly 7.2 to 8.7 based
   on RMSE alone. No formal calibrated prediction interval was computed. Generated molecules with
   novel scaffolds may fall outside the oracle's applicability domain.

2. **BBB model accuracy:** Test R² = 0.354, RMSE = 0.621 logBB units, trained on 742 molecules
   from B3DB (CNS drugs and probe compounds). BACE1 peptidomimetic scaffolds are structurally
   distinct from B3DB training molecules. Applicability domain is unverified. logBB predictions
   for BACE1 candidates carry high uncertainty. A molecule predicted at logBB = −0.014 is
   statistically indistinguishable from logBB = 0.0 given RMSE = 0.621.

3. **Docking not run in this pilot.** Only oracle and BBB model predictions were computed.
   No AutoDock Vina docking was performed on the 50k pilot molecules. Docking results from
   the 16k run are not extended to this pool.

4. **All results are computational only.** No experimental validation (synthesis, in vitro BACE1
   IC50 assay, PAMPA BBB, MDCK-MDR1) has been performed on any molecule in this pilot.

5. **Seed 42 matches the 16k run but this is NOT an exact replay.** The two runs used different
   numbers of batches, producing overlapping but non-identical molecule pools. Statistics are
   comparable but not directly subtractive.

6. **Condition non-responsiveness.** The generator's mean predicted pIC50 (5.48) is far below
   the requested pIC50 = 7.0, consistent with all prior conditioning experiments. The condition
   influences generation but does not reliably shift the oracle-predicted property distribution.

---

*Report generated from verified file contents. All numerical claims are sourced from
`results/baseline_screen_50000_summary.json` and `results/baseline_screen_50000_comparison.csv`.*
