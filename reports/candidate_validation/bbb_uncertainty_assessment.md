# BBB Prediction Uncertainty Assessment

## BBB Model Facts

- Algorithm: ExtraTrees regressor
- Training set: B3DB quantitative logBB dataset
  - Train: 742 molecules (512 scaffolds)
  - Scaffold-split test: 139 molecules
- Test R² = 0.3538
- Test RMSE = 0.6215 logBB units
- Test MAE = 0.4836 logBB units
- B3DB logBB mean = −0.0779, std = 0.7516

## Domain Warning: BACE1 Scaffolds Are Out-of-Domain

The B3DB training set consists primarily of CNS drugs and CNS-active small molecules.
BACE1 inhibitors are peptidomimetics with iminopyrimidinone/amidine warheads and
extended P1'–P2' side-chains that are structurally dissimilar from typical CNS drugs.
Morgan ECFP4 Tanimoto similarities between the 6 candidates and the B3DB training set
are computed below. Any candidate with max_B3DB_Tc < 0.4 is formally outside the
applicability domain.

## RMSE Interpretation Warning

The RMSE (0.621 logBB units) is a **root-mean-squared error across a held-out test set**.
It is a typical error scale indicator, NOT a 95% confidence interval.
No saved test-set predictions are available, so calibrated per-molecule intervals
cannot be computed. The ± RMSE figures below are heuristic uncertainty bounds only.

## Per-Candidate BBB Assessment

### Candidate #1

- Predicted logBB: -0.0202
- Distance from threshold (0.0): 0.0202 logBB units = 0.032 RMSE-units
- max_B3DB_Tc=0.173; applicability_domain=OUTSIDE
- **Verdict: INDETERMINATE**

  Justification: 
  Predicted logBB = -0.0202 is within 1 RMSE (0.621) of the 0.0 threshold. Given the model's limited accuracy (R²=0.354), out-of-domain scaffolds, and the absence of calibrated uncertainty, the prediction straddles the threshold and cannot be reliably classified as BBB+ or BBB−.

### Candidate #3

- Predicted logBB: -0.0324
- Distance from threshold (0.0): 0.0324 logBB units = 0.052 RMSE-units
- max_B3DB_Tc=0.183; applicability_domain=OUTSIDE
- **Verdict: INDETERMINATE**

  Justification: 
  Predicted logBB = -0.0324 is within 1 RMSE (0.621) of the 0.0 threshold. Given the model's limited accuracy (R²=0.354), out-of-domain scaffolds, and the absence of calibrated uncertainty, the prediction straddles the threshold and cannot be reliably classified as BBB+ or BBB−.

### Candidate #5

- Predicted logBB: -0.2242
- Distance from threshold (0.0): 0.2242 logBB units = 0.361 RMSE-units
- max_B3DB_Tc=0.220; applicability_domain=OUTSIDE
- **Verdict: LIKELY_NEGATIVE**

  Justification: 
  Predicted logBB = -0.2242 is 0.36 RMSE-units below 0. While still within 1 RMSE, the more negative value makes a true positive logBB less likely. Given out-of-domain scaffolds, classification remains uncertain but leans negative.

### Candidate #6

- Predicted logBB: -0.0142
- Distance from threshold (0.0): 0.0142 logBB units = 0.023 RMSE-units
- max_B3DB_Tc=0.186; applicability_domain=OUTSIDE
- **Verdict: INDETERMINATE**

  Justification: 
  Predicted logBB = -0.0142 is within 1 RMSE (0.621) of the 0.0 threshold. Given the model's limited accuracy (R²=0.354), out-of-domain scaffolds, and the absence of calibrated uncertainty, the prediction straddles the threshold and cannot be reliably classified as BBB+ or BBB−.

## Summary Table

| Candidate | Predicted logBB | |logBB|/RMSE | Applicability Domain | Verdict |
|-----------|-----------------|-------------|----------------------|---------|
| #1 | -0.0202 | 0.032 | OUTSIDE (max_Tc=0.173) | INDETERMINATE |
| #3 | -0.0324 | 0.052 | OUTSIDE (max_Tc=0.183) | INDETERMINATE |
| #5 | -0.2242 | 0.361 | OUTSIDE (max_Tc=0.220) | LIKELY_NEGATIVE |
| #6 | -0.0142 | 0.023 | OUTSIDE (max_Tc=0.186) | INDETERMINATE |

## Note on Threshold

The original logBB >= 0.0 threshold has NOT been changed. All 6 candidates have
predicted logBB < 0.0. The INDETERMINATE label means the prediction cannot reliably
confirm OR deny BBB penetrance — it does NOT mean the threshold has been passed.

Content was rephrased for compliance with licensing restrictions.