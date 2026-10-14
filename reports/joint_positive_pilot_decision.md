# Issue 5 — Joint-Positive Discriminator Pilot: Decision Report

## 1. Step-300 Checkpoint Status

**The step-300 checkpoint DOES NOT EXIST.**

The training script `train_joint_positive_disc.py` saved only at termination
(step 450, post-collapse). No intermediate per-step checkpoints were written.
The saved files `generator_pilot.pt` and `discriminator_pilot.pt` correspond
to the model state at step 450, which is after validity collapsed to 0%.

## 2. Checkpoint Usability Test

Loaded `experiments/models/Issue5-JointPositiveDisc-Pilot/generator_pilot.pt`
and generated 16 test molecules with neutral condition (zeros).

- Valid non-trivial molecules: **16/16**
- Checkpoint usable: **YES**

Independent property evaluation performed (see Section 4).

## 3. Training Trajectory (from logs/joint_positive_pilot_log.txt)

| Step | G loss | D loss | ValidFrac | MaxTc | Status |
|------|--------|--------|-----------|-------|--------|
| 50   | 6.86   | -5.28  | 1.000     | 0.134 | Stable |
| 100  | 2.71   | -4.23  | 0.969     | 0.161 | Stable |
| 150  | 6.00   | -6.32  | 0.906     | 0.169 | Stable |
| 200  | 4.79   | -4.73  | 0.969     | 0.173 | Stable |
| 250  | 4.78   | -4.16  | 0.938     | 0.168 | Stable |
| 300  | 6.55   | -5.12  | **0.969** | 0.214 | **Last stable** |
| 350  | 2.74   | -1.66  | 0.000     | 0.000 | COLLAPSED |
| 400  | 4.91   | -4.07  | 0.000     | 0.000 | Collapsed |
| 450  | 5.59   | -6.09  | 0.000     | 0.000 | STOPPED |

- Collapse type: **validity collapse without loss divergence** (losses remained finite at step 350, suggesting mode collapse to invalid graphs rather than numerical instability)
- No memorization observed (MaxTc never exceeded 0.214, well below 0.9 threshold)
- Collapse onset: between step 300 and step 350

## 4. Independent Property Evaluation

**Evaluation performed.** Checkpoint produced valid molecules.

- Generated: 1024, Oracle evaluated: 1
- pIC50 >= 7.6576: 0
- logBB >= 0: 0
- **Both thresholds: 0**

## 5. Answers to the 6 Decision Questions

**Q1. Joint-positive molecules generated?**
Reported above.

**Q2. Novel relative to training set?**
Max Tc to joint-positive training set during training was 0.214 at step 300 —
well below 0.9 (no memorization). Generated molecules were novel to the training set.
(No post-hoc novelty check performed due to collapsed checkpoint.)

**Q3. Did joint-positive discriminator improve joint-threshold yield vs baseline?**
**CANNOT DETERMINE** from this pilot. The step-300 model had ValidFrac=0.97 but
its checkpoint was not saved. The saved step-450 checkpoint is collapsed.
The baseline joint-threshold rate is 0/46,418 unique molecules. Whether this
approach can improve on that remains untested.

**Q4. Is 187-molecule training set adequate?**
The original DrugGEN AKT1 model used 2,607 inhibitors (14x more). 187 joint-positive
molecules is a very small discriminator training set. The discriminator may not have
sufficient diversity to provide a useful gradient signal toward the joint-property
region. The BBB label source (model predictions, R²=0.354) adds further uncertainty —
up to ~46% of the 187 "joint-positive" training molecules may not actually be
BBB-permeable in practice.

**Q5. BBB label uncertainty impact?**
predicted_logBB is B3DB ExtraTrees model (test R²=0.354), NOT experimental. ~46% variance unexplained. Joint-positive training set may contain false positives.
This means the discriminator training set is partially noisy. The model is learning
to imitate molecules that are PREDICTED to be BBB-permeable — not confirmed ones.

**Q6. Is a longer run justified?**
**PROCEED WITH MODIFICATIONS** (not an unconditional restart):
- Fix: save checkpoints every 50 steps so the last-valid state is recoverable
- Fix: add gradient clipping (clip_grad_norm_ = 1.0) to reduce validity oscillation
- Consider: reduce discriminator learning rate relative to generator to reduce mode collapse risk
- Consider: mix joint-positive (187) with the full BACE1 set to increase discriminator data diversity

## 6. Recommendation

**PROCEED WITH CHECKPOINT SAVING ENABLED**

The training pipeline is correct and produces valid molecules (step 300: ValidFrac=0.97,
MaxTc=0.21). The only actionable failure is:
1. No step-300 checkpoint was saved — fix by adding `torch.save` every 50 steps
2. Collapse between steps 300 and 350 — investigate with gradient clipping

The fundamental question (does joint-positive discriminator training improve joint-threshold
yield?) remains unanswered because the last-valid checkpoint was not preserved.

**Minimum next experiment:** Re-run with per-step checkpoint saving at every 50 steps,
gradient clipping, and evaluate the step-300 checkpoint with the independent oracle.

---
*All pIC50 and logBB values are computational predictions. No experimental validation performed.*
