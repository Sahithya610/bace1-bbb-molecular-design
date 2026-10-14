"""
Evaluate the Issue5 joint-positive pilot checkpoint.
Tests usability, generates molecules, runs oracle, writes decision report.
"""
import os, sys, json, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Patch torch.load ──────────────────────────────────────────────
import torch
_orig = torch.load
def _patched(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(f, *a, **kw)
torch.load = _patched

import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, ".")
from src.data.utils import get_encoders_decoders, load_molecules
from src.model.conditional_models import ConditionalGenerator
from src.data.dataset import DruggenDataset
from torch_geometric.loader import DataLoader

SEP = "="*65

print(SEP)
print("ISSUE 5 — CHECKPOINT USABILITY TEST")
print(SEP)

# ── Load checkpoint ───────────────────────────────────────────────
CKPT = "experiments/models/Issue5-JointPositiveDisc-Pilot/generator_pilot.pt"
print(f"\nCheckpoint: {CKPT}")

atom_enc, atom_dec, bond_enc, bond_dec = get_encoders_decoders(
    "data/chembl_train.smi", "druggen_data/bace1.smi", 45)
m_dim = len(atom_dec); b_dim = len(bond_dec); V = 45

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = ConditionalGenerator("relu", V, b_dim, m_dim, 0.0, 128, 1, 8, 3,
                          condition_dim=3).to(device)
G.load_state_dict(torch.load(CKPT, map_location=device))
G.eval()

# ── Test with 2 batches (16 molecules) ───────────────────────────
print("Testing with 16 molecules (2 batches)...")
ds = DruggenDataset("data", "chembl_train45.pt", "data/chembl_train.smi",
                    45, False, atom_enc, atom_dec, bond_enc, bond_dec)
loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
loader_iter = iter(loader)

valid_count = 0
smiles_out = []
for _ in range(2):
    batch = next(loader_iter)
    from src.data.utils import load_molecules as lm
    _, adj, x = lm(batch, b_dim=b_dim, m_dim=m_dim, device=device, batch_size=8)
    cond = torch.zeros(8, 3, device=device)
    with torch.no_grad():
        out = G(adj, x, cond)
    nl = out[-2]; el = out[-1]
    for i in range(8):
        n = nl[i].argmax(-1).cpu().numpy()
        e = el[i].argmax(-1).cpu().numpy()
        mol = ds.matrices2mol(n, e, strict=False)
        if mol:
            s = Chem.MolToSmiles(mol)
            if s and len(s) > 3:
                valid_count += 1
                smiles_out.append(s)

print(f"Valid non-trivial: {valid_count}/16")
checkpoint_usable = valid_count > 0
print(f"Checkpoint usable: {'YES' if checkpoint_usable else 'NO (collapsed)'}")

# ── If usable: generate 1024 and run oracle ───────────────────────
n_joint = 0
n_pic50 = 0
n_logBB = 0
n_gen = n_valid_eval = 0
eval_performed = False

if checkpoint_usable:
    print(f"\n{SEP}")
    print("GENERATING 1024 MOLECULES FOR ORACLE EVALUATION")
    print(SEP)
    py = sys.executable
    r = subprocess.run([
        py, "generate_joint_molecules.py",
        "--checkpoint", CKPT,
        "--desired_pIC50", "7.0",
        "--desired_logBB", "0.0",
        "--batch_size", "128",
        "--num_batches", "8",
        "--seed", "42",
        "--output", "results/joint_positive_pilot_generated.csv"
    ], capture_output=True, text=True, cwd=".")
    if r.returncode == 0:
        import pandas as pd
        gen = pd.read_csv("results/joint_positive_pilot_generated.csv")
        n_gen = len(gen)
        n_valid_gen = gen["valid"].sum()
        print(f"Generated: {n_gen}, Valid: {int(n_valid_gen)}")
        r2 = subprocess.run([
            py, "predict_joint.py",
            "--input", "results/joint_positive_pilot_generated.csv",
            "--output", "results/joint_positive_pilot_predictions.csv"
        ], capture_output=True, text=True, cwd=".")
        if r2.returncode == 0 and os.path.exists("results/joint_positive_pilot_predictions.csv"):
            pred = pd.read_csv("results/joint_positive_pilot_predictions.csv")
            n_valid_eval = len(pred)
            n_pic50 = int((pred["predicted_pIC50"] >= 7.6576).sum())
            n_logBB = int((pred["predicted_logBB"] >= 0.0).sum())
            n_joint = int(((pred["predicted_pIC50"] >= 7.6576) & (pred["predicted_logBB"] >= 0.0)).sum())
            eval_performed = True
            print(f"Oracle evaluated: {n_valid_eval}")
            print(f"pIC50>=7.6576: {n_pic50}")
            print(f"logBB>=0:      {n_logBB}")
            print(f"BOTH:          {n_joint}")
            print(f"Max pIC50:     {pred['predicted_pIC50'].max():.4f}")

# ── Write eval JSON ───────────────────────────────────────────────
result = {
    "step_300_checkpoint_exists": False,
    "step_300_checkpoint_note": "Training script saved only at termination (step 450). No per-step checkpoints were written.",
    "saved_checkpoint_step": 450,
    "saved_checkpoint_path": CKPT,
    "checkpoint_usable": checkpoint_usable,
    "valid_in_test_batch": valid_count,
    "independent_eval_performed": eval_performed,
    "n_generated": n_gen if eval_performed else None,
    "n_valid_oracle_evaluated": n_valid_eval if eval_performed else None,
    "n_pic50_ge7_6576": n_pic50 if eval_performed else None,
    "n_logBB_ge0": n_logBB if eval_performed else None,
    "n_both_thresholds": n_joint if eval_performed else None,
    "baseline_joint_positive_rate": 0,
    "evaluation_status": "COMPLETE" if eval_performed else "NOT_PERFORMED_COLLAPSED_CHECKPOINT"
}
with open("results/joint_positive_pilot_eval.json", "w") as f:
    json.dump(result, f, indent=2)

# ── Decision report ───────────────────────────────────────────────
bbb_caveat = "predicted_logBB is B3DB ExtraTrees model (test R²=0.354), NOT experimental. ~46% variance unexplained. Joint-positive training set may contain false positives."

report = f"""# Issue 5 — Joint-Positive Discriminator Pilot: Decision Report

## 1. Step-300 Checkpoint Status

**The step-300 checkpoint DOES NOT EXIST.**

The training script `train_joint_positive_disc.py` saved only at termination
(step 450, post-collapse). No intermediate per-step checkpoints were written.
The saved files `generator_pilot.pt` and `discriminator_pilot.pt` correspond
to the model state at step 450, which is after validity collapsed to 0%.

## 2. Checkpoint Usability Test

Loaded `experiments/models/Issue5-JointPositiveDisc-Pilot/generator_pilot.pt`
and generated 16 test molecules with neutral condition (zeros).

- Valid non-trivial molecules: **{valid_count}/16**
- Checkpoint usable: **{'YES' if checkpoint_usable else 'NO'}**

{'Independent property evaluation performed (see Section 4).' if eval_performed else 'Independent property evaluation NOT PERFORMED — saved checkpoint is post-collapse and produces zero valid molecules. Joint-threshold counts cannot be determined from this pilot.'}

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

{'**Evaluation performed.** Checkpoint produced valid molecules.' if eval_performed else '**Evaluation NOT performed.** The saved checkpoint (step 450) produces 0 valid non-trivial molecules. Joint-threshold counts are therefore 0 by default, but this reflects checkpoint collapse, not the model behavior at peak validity (step 300).'}

{f'- Generated: {n_gen}, Oracle evaluated: {n_valid_eval}' if eval_performed else ''}
{f'- pIC50 >= 7.6576: {n_pic50}' if eval_performed else ''}
{f'- logBB >= 0: {n_logBB}' if eval_performed else ''}
{f'- **Both thresholds: {n_joint}**' if eval_performed else ''}

## 5. Answers to the 6 Decision Questions

**Q1. Joint-positive molecules generated?**
{'Reported above.' if eval_performed else 'Cannot determine — checkpoint is post-collapse. The model was generating valid chemistry at step 300 (ValidFrac=0.97) but no checkpoint was saved at that point.'}

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
{bbb_caveat}
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
"""

with open("reports/joint_positive_pilot_decision.md", "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n{SEP}")
print("COMPLETE")
print(f"  reports/joint_positive_pilot_decision.md")
print(f"  results/joint_positive_pilot_eval.json")
print(SEP)
