# Methodological Limitations and Prioritised Next Experiments

**Project:** DrugGEN BACE1/BBB Dual-Target Validation  
**Deliverable:** Methodological Audit and Experimental Roadmap  
**Date:** October 2026  

---

## 1. Critical Methodological Limitations

### A. The BBB Permeability Predictor Generalisation Gap
1. **Low Variance Explained ($R^2 = 0.354$):**  
   The B3DB ExtraTrees regression model achieves a scaffold-split test $R^2$ of only **0.3538** and an **$\text{RMSE} = 0.6215$ log units**. Approximately 65% of variance in experimental blood-brain barrier partitioning remains unexplained.
2. **Extreme Borderline Fragility of $\text{logBB} \ge 0.0$ Hits:**  
   - `ANALOGUE_0011` has a predicted logBB of **+0.0139** (margin: $+0.0139$).
   - `Candidate_1308` has a predicted logBB of **+0.0227** (margin: $+0.0227$).
   - `Candidate_1902` has a predicted logBB of **+0.0904** (margin: $+0.0904$).  
   All three margins are **7 to 45 times smaller** than the model's test RMSE ($0.6215$). Furthermore, individual decision tree predictions within the 500-tree ensemble disperse with a standard deviation of $\pm 0.58$ log units. A 95% prediction interval around $+0.0139$ spans approximately $[-1.20, +1.23]$.
3. **Out-of-Domain Scaffolds:**  
   The B3DB training set comprises CNS drugs and small probes (mean $\text{MW} = 310$ g/mol). BACE1 inhibitor candidates (peptidomimetic, spiro-oxazoline, or multi-ring architectures with $\text{MW} = 380 - 435$ g/mol) have low structural similarity to B3DB ($T_c \le 0.22$). They operate near the edge of the model's applicability domain.
4. **Mandatory Standard:**  
   A predicted logBB value just above 0.0 **cannot and must not be reported as confirmed in vivo brain penetration**.

### B. BACE1 ExtraTrees Oracle Uncertainty ($R^2 = 0.616$, $\text{RMSE} = 0.755$)
1. **Error Margin Comparison:**  
   While the BACE1 oracle is more accurate than the BBB model ($R^2 = 0.6156$, test $\text{RMSE} = 0.7553$), `ANALOGUE_0011`'s predicted pIC50 margin above threshold ($7.7983 - 7.6576 = +0.1407$) is **5.4 times smaller** than the test RMSE.
2. **Ensemble Variance:**  
   Across the 500 estimators, tree predictions for `ANALOGUE_0011` range from 4.78 to 8.99 with a standard deviation of 0.93. The prediction provides solid qualitative prioritization but does not guarantee laboratory nanomolar potency.

### C. Molecular Docking Score Interpretation
1. **Vina Scores are NOT Measured Free Energies:**  
   AutoDock Vina scores are empirical physics-based scoring function approximations, not rigorous free energy calculations ($\Delta G_{\text{bind}}$) or inhibition constants ($K_i$).
2. **Relative Ranking Utility Only:**  
   While redocking of reference ligand 3BN succeeded cleanly ($\text{RMSD} = 1.33\text{ \AA} < 2.0\text{ \AA}$), validating the search space and scoring geometry, docking scores cannot substitute for biochemical binding measurements.

---

## 2. Concrete, Prioritised Experimental Roadmap

To transition these in-silico hypotheses into verified biomedical reality, the following experimental sequence is required:

### Step 1: In Vitro Enzymatic Potency Assay (Priority 1)
- **Assay Type:** Recombinant human BACE1 fluorescence resonance energy transfer (FRET) cleavage assay (substrate: $\text{Rh-EVNLDAEFK-Quencher}$).
- **Target Threshold:** Measured $\text{IC}_{50} \le 22.0\text{ nM}$ (equivalent to $\text{pIC50} \ge 7.6576$).
- **Candidates to Test:** `ANALOGUE_0011`, `Candidate_1308`, and `Candidate_1902`.
- **Expected Outcome:** Validates whether `ANALOGUE_0011` retains nanomolar catalytic dyad inhibition as predicted by the oracle ($7.7983$).

### Step 2: Blood-Brain Barrier Permeability & Efflux Profiling (Priority 2)
- **Assays:**
  1. **Parallel Artificial Membrane Permeability Assay for the Blood-Brain Barrier (PAMPA-BBB):** Measures passive transcellular permeability ($P_{\text{e}}$). Compounds with $P_{\text{e}} > 4.0 \times 10^{-6}\text{ cm/s}$ are classified as $\text{CNS+}$.
  2. **MDCK-MDR1 Bidirectional Cell Permeability Assay:** Measures apical-to-basolateral ($P_{\text{app}, A \to B}$) and basolateral-to-apical ($P_{\text{app}, B \to A}$) transport across Madin-Darby canine kidney cells expressing human P-glycoprotein (P-gp).
- **Target Criteria:** Efflux ratio $< 2.5$ (rules out active P-gp efflux liability).
- **Goal:** Definitively resolves whether the borderline predicted logBB values ($+0.014$ to $+0.090$) translate to real-world central nervous system exposure.

### Step 3: High-Level In Silico Molecular Dynamics (MD) & Free Energy Perturbation (Priority 3)
- **Method:** 100-ns all-atom explicit-solvent (TIP3P) MD simulation in AMBER or GROMACS with CHARMM36m/GAFF2 force fields.
- **Goals:**
  - Track hydrogen-bond occupancy of catalytic dyad Asp32/Asp228 and the crystallographic flap water with `ANALOGUE_0011`.
  - Calculate relative binding free energies ($\Delta \Delta G$) using alchemical Free Energy Perturbation (FEP) between `PARENT_A1` and `ANALOGUE_0011`.

---
*All reported values remain computational models until validated by experimental wet-lab assays.*
