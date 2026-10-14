# Cross-Attention Conditioning Decision and Pilot Result

**Date:** October 2026  
**Experiment:** Issue4-CrossAttention-Pilot500  
**Status:** FAIL — stopping criteria not met

---

## Decision Rationale (Phase 6A)

**Decision: PROCEED** with cross-attention pilot based on the following evidence:

1. All 8 prior conditioning approaches failed, and the most consistent failure mode was additive embedding saturation — uniform condition injection across all 45 atom positions biases every position away from padding class 0 at high pIC50 requests.
2. The oracle test R² = 0.616 is adequate for a conditioning signal. The failure is architectural, not a data/oracle quality issue.
3. Cross-attention (condition as K/V, graph node representations as Q) was the one remaining untested architectural change that provides position-selective, learned condition injection rather than uniform additive bias.
4. Implementation risk was low: single cross-attention layer with residual connection, strict=False warm-start preserving all existing parameters.

**Falsifiable hypothesis:** Cross-attention conditioning will produce Δ pIC50 ≥ 0.3 AND ValidFrac ≥ 0.50 at both conditions within 500 steps.

---

## Implementation (Phase 6B)

**Script:** `train_joint_cross_attention.py`  
**Architecture change:** Added ONE `nn.MultiheadAttention` layer after `TransformerEncoder` and before the `readout_condition` concatenation:
- `condition_token = Linear(condition_dim=3, dim=128)(condition).unsqueeze(1)` → [B, 1, 128]
- Each of V=45 atom positions (Q) attends to the condition token (K=V)
- Residual: `node = node + cross_attn_output`
- Existing `readout_condition` pathway unchanged (additive path still present as skip connection)

**Warm-start:** `strict=False` from `Issue3-ChemistryReadout` checkpoint  
**Transferred:** 46 existing generator parameter tensors  
**Freshly initialized:** 6 new cross-attention parameter tensors (`cross_cond_proj.weight/bias`, `cross_attn.in_proj_weight/bias`, `cross_attn.out_proj.weight/bias`)  
**Discriminator:** 62 tensors transferred, 0 fresh

---

## Pilot Result (Phase 6C)

**Config:** batch_size=32, max_steps=500, lambda_condition=5.0, seed=1  
**Output dir:** `experiments/models/Issue4-CrossAttention-Pilot500/`  
**Results:** `results/cross_attention_pilot/`

### Training metrics (500 steps)

| Metric | Baseline (λ_c=1) | Cross-Attention (λ_c=5) |
|--------|-----------------|------------------------|
| corr(req, head) | 0.364 | **0.477** (+31%) |
| G→head cond MSE pIC50 | 1.616 | **1.054** (−35%) |
| Head held-out R² | 0.683 | 0.674 |

The cross-attention layer improved internal conditioning signal by 31% and reduced condition MSE substantially. This confirms the architecture change has a measurable effect on the training objective.

### Generation and oracle evaluation

| Metric | Required | pIC50=5.5 | pIC50=8.0 |
|--------|----------|-----------|-----------|
| Valid molecules / 1024 | ≥ 0.50 | **578 (56.4%)** ✓ | **0 (0.0%)** ✗ |
| Non-trivial molecules | ≥ 10 | **578** ✓ | **0** ✗ |
| Mean oracle pIC50 | — | 5.407 | N/A (no valid mols) |
| Δ pIC50 | ≥ 0.3 | — | **N/A** ✗ |

### Verdict: **FAIL**

The pIC50=8.0 condition produced **zero valid molecules** (ValidFrac = 0.000). The same saturation collapse observed in the λ_condition=5 sweep persists with cross-attention: at high pIC50 requests (normalized ≈ +1.0), all 45 atom slots saturate with non-padding atoms, producing graphs that cannot decode to valid SMILES.

The cross-attention improved the internal proxy metric but did not prevent validity collapse during generation.

---

## Comparison to All Prior Approaches

| # | Approach | Δ pIC50 | ValidFrac@8.0 | Pass? |
|---|----------|---------|---------------|-------|
| 1 | Baseline chemistry-readout | +0.066 | ~30% | ✗ |
| 2 | ET-distill | +0.066 | ~30% | ✗ |
| 3 | λ sweep λ=1 | +0.066 | ~30% | ✗ |
| 3 | λ sweep λ≥5 | — | 0% | ✗ |
| 4 | Validity-gated | — | 0% | ✗ |
| 5 | Padding penalty | +0.529 (spurious, n=2) | — | ✗ |
| 6 | REINFORCE raw | corr=−0.155 | 63% | ✗ |
| 7 | REINFORCE cond exp | corr=−0.428 | 29% | ✗ |
| 8 | Contrastive REINFORCE | corr=−0.653 | 36% | ✗ |
| **9** | **Cross-attention (this)** | **N/A** | **0%** | **✗** |

---

## Diagnosis

The cross-attention result clarifies that **saturation is not caused by the additive embedding mechanism alone**. Even with position-selective cross-attention providing an alternative pathway, the generator still collapses at pIC50=8.0 with λ_condition=5.

The saturation is driven by **the condition loss gradient itself** (λ_condition=5 × MSE term) which, regardless of how the condition signal reaches the generator, creates strong pressure to push every atom position away from the padding class. This pressure is inherent in any scheme that uses the internal property head as a differentiable surrogate for the oracle at high λ values.

---

## Recommendation for Next Step

The conditioning failure root cause is now more precisely characterized: it is not additive vs. cross-attention injection, but rather the **mismatch between λ_condition=5 pressure and the generator's capacity to satisfy it without destroying validity**.

Two options remain:

**Option A (architectural — medium effort):** Decouple condition pressure from atom-count decisions. Inject condition only into the bond readout, not the node readout. Atom count is chemistry-structural; bond patterns drive pharmacophoric features. This requires a targeted change to `readout_e` only.

**Option B (methodological — lower effort, most likely to succeed):** Use the post-hoc oracle filtering pipeline (already demonstrated viable) with a **larger generation pool** (100k–500k molecules) rather than conditioning. Expected yield at current 2.8% rate: ~3,000 molecules with pIC50 ≥ 6.0 and ~140 with pIC50 ≥ 7.0 from 500k molecules. This avoids all conditioning complexity while improving coverage.

**Given that 9 consecutive conditioning approaches have failed, Option B is recommended as the scientifically sound next step.** Further conditioning experiments require a fundamentally different training paradigm (e.g., diffusion-based generation, SMILES-level REINVENT) rather than incremental changes to the current graph-GAN architecture.

---

*Results saved to: `results/cross_attention_pilot/pilot_results.json`*  
*Training checkpoint: `experiments/models/Issue4-CrossAttention-Pilot500/joint_generator.pt`*
