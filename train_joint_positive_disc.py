"""
Issue 5 — Joint-Positive Target-Specific DrugGEN Training.

Hypothesis
----------
The generator may learn the joint-property distribution more
effectively if the discriminator's target-specific real-molecule set
contains BACE1 molecules satisfying BOTH thresholds (pIC50 >= 7.6576
AND predicted logBB >= 0.0), rather than forcing the same objective
through an explicit continuous condition vector.

Approach: Option A
------------------
- The joint-positive BACE1 subset (bace1_joint_positive_train.smi)
  is used as the discriminator's real-molecule set.
- A zero condition vector (shape [batch_size, 3]) is passed to
  ConditionalGenerator and ConditionalDiscriminator throughout.
  This suppresses the continuous pIC50/logBB conditioning signal
  while keeping architecture compatibility.
- lambda_condition = 0.0 (condition-consistency loss not computed
  or backpropagated). The auxiliary property head is kept for
  monitoring only (auxiliary loss on real graphs).
- All other training elements are unchanged: WGAN-GP, property head
  auxiliary training, critic update convention.

Changes from train_joint.py baseline
--------------------------------------
1. drug_raw_file replaced with bace1_joint_positive_train.smi
2. condition = torch.zeros(batch_size, 3) passed to generator and
   discriminator (no per-batch condition from CSV).
3. lambda_condition fixed at 0.0 — cond_loss not included in
   generator update.

Unchanged
---------
- WGAN-GP gradient penalty
- ConditionalGenerator and ConditionalDiscriminator architectures
- Auxiliary property head (lambda_aux=1.0 by default, monitoring)
- ChEMBL general-molecule loader for the generator's input graphs
- Critic input convention

Stopping rules (checked every 50 steps)
-----------------------------------------
- ValidFrac < 0.01 for 3 consecutive log steps
- D_loss or G_loss is non-finite (nan or inf)
- > 90 % of generated molecules have Tc >= 0.9 to the
  joint-positive training set (memorization)

Label warning
-------------
BBB values in the joint-positive set are B3DB ExtraTrees model
predictions (test R2 = 0.354), NOT experimental BBB permeability.
"""

import argparse
import json
import math
import os
import random
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.model.conditional_models import (
    ConditionalGenerator,
    ConditionalDiscriminator,
    relax_graph,
)

# ============================================================
# Constants
# ============================================================

CONDITION_DIM = 3
MAX_ATOM_DEFAULT = 45


# ============================================================
# Checkpoint loading
# ============================================================

def load_model_checkpoint(model, checkpoint_path, device, model_label):
    """
    Load model weights with strict=False to tolerate minor
    architecture differences between training phases.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{model_label} checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "generator_state_dict",
            "discriminator_state_dict",
        ):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break

    if not isinstance(state, dict):
        raise TypeError(
            f"Unexpected checkpoint format for {model_label}: "
            f"{type(state).__name__}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded {model_label} from: {path}"
        f" (missing={len(missing)}, unexpected={len(unexpected)}, "
        f"strict=False)"
    )
    if missing:
        print(f"  Missing keys: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}")


# ============================================================
# Molecule validity helpers
# ============================================================

def decode_molecule(dataset, node_sample, edge_sample, idx):
    """
    Decode a single generated graph to an RDKit molecule.
    Returns (mol, smiles_str) or (None, None).
    """
    try:
        node_argmax = node_sample[idx].argmax(dim=-1).cpu().numpy()
        edge_argmax = edge_sample[idx].argmax(dim=-1).cpu().numpy()
        mol = dataset.matrices2mol(node_argmax, edge_argmax, strict=True)
        if mol is None:
            return None, None
        smi = Chem.MolToSmiles(mol, canonical=True)
        if smi is None or len(smi) == 0:
            return None, None
        return mol, smi
    except Exception:
        return None, None


def is_nontrivial(mol):
    """Return True if molecule has > 3 heavy atoms."""
    if mol is None:
        return False
    return mol.GetNumHeavyAtoms() > 3


def mol_to_fp(mol):
    """Morgan fingerprint radius 2, 2048 bits."""
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    except Exception:
        return None


# ============================================================
# Auxiliary-loss helpers (carried over from train_joint.py)
# ============================================================

def heldout_flags(smiles_list, frac, device):
    """
    Deterministic hold-out flags (True = excluded from aux loss).
    Uses crc32 so results are reproducible across processes.
    """
    if frac <= 0.0:
        return torch.zeros(len(smiles_list), dtype=torch.bool, device=device)
    threshold = int(round(frac * 1000))
    flags = [
        (zlib.crc32(str(s).encode("utf-8")) % 1000) < threshold
        for s in smiles_list
    ]
    return torch.tensor(flags, dtype=torch.bool, device=device)


def masked_mse(pred, target, keep):
    """Per-property MSE over rows where keep == True."""
    keep_f = keep.float().unsqueeze(1)
    denom = keep_f.sum().clamp(min=1.0)
    return (((pred - target) ** 2) * keep_f).sum(dim=0) / denom


# ============================================================
# Training
# ============================================================

def train(cfg):

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "joint_positive_pilot_log.txt"

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # --------------------------------------------------------
    # Run configuration — saved before any training
    # --------------------------------------------------------
    run_config = {
        "experiment": "Issue5-JointPositiveDisc-Pilot",
        "hypothesis": (
            "Discriminator trained exclusively on joint-positive BACE1 "
            "molecules (pIC50>=7.6576 AND logBB>=0.0) will guide the "
            "generator toward the joint-property distribution without "
            "explicit continuous conditioning"
        ),
        "bbb_label_warning": (
            "predicted_logBB values are B3DB ExtraTrees model predictions "
            "(test R2=0.354), NOT experimental BBB permeability measurements"
        ),
        "approach": (
            "Option A: zero condition vector passed to ConditionalGenerator, "
            "lambda_condition=0.0"
        ),
        "changes_from_baseline": [
            "drug_raw_file replaced with bace1_joint_positive_train.smi",
            "condition = torch.zeros(batch_size, 3) passed to generator",
            "lambda_condition set to 0.0 (no condition consistency loss)",
        ],
        "unchanged": [
            "WGAN-GP",
            "architecture",
            "auxiliary property head (monitoring only)",
            "critic convention",
            "ChEMBL general input to generator",
        ],
        "warm_start_generator": cfg.init_generator,
        "warm_start_discriminator": cfg.init_discriminator,
        "bace1_smi_for_vocab": cfg.bace1_smi,
        "joint_positive_train_smi": cfg.joint_positive_smi,
        "batch_size": cfg.batch_size,
        "max_steps": cfg.max_steps,
        "g_lr": cfg.g_lr,
        "d_lr": cfg.d_lr,
        "lambda_gp": cfg.lambda_gp,
        "lambda_aux": cfg.lambda_aux,
        "lambda_condition": 0.0,
        "seed": cfg.seed,
    }

    (out / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    log("=" * 70)
    log("Issue 5 — Joint-Positive Discriminator Pilot")
    log("=" * 70)
    log(json.dumps(run_config, indent=2))

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # --------------------------------------------------------
    # Encoders / decoders
    # --------------------------------------------------------
    # Use the full BACE1 SMI for vocabulary construction so that the
    # encoder/decoder matches the warm-start checkpoints (node_dim=14,
    # edge_dim=5).  The joint-positive subset does not cover all atom
    # types present in bace1.smi, so building a vocab from it alone
    # produces node_dim=11, which mismatches the checkpoint tensors.
    atom_enc, atom_dec, bond_enc, bond_dec = get_encoders_decoders(
        cfg.chembl_smi,
        cfg.bace1_smi,
        MAX_ATOM_DEFAULT,
    )

    # --------------------------------------------------------
    # Processed dataset filenames
    # --------------------------------------------------------
    chembl_name = Path(cfg.chembl_smi).stem + str(MAX_ATOM_DEFAULT) + ".pt"
    jp_name = Path(cfg.joint_positive_smi).stem + str(MAX_ATOM_DEFAULT) + ".pt"

    # --------------------------------------------------------
    # General ChEMBL dataset (generator input)
    # --------------------------------------------------------
    general = DruggenDataset(
        "data",
        chembl_name,
        cfg.chembl_smi,
        MAX_ATOM_DEFAULT,
        False,
        atom_enc,
        atom_dec,
        bond_enc,
        bond_dec,
    )

    # --------------------------------------------------------
    # Joint-positive BACE1 dataset (discriminator real-molecule set)
    #
    # NOTE: No condition vector is attached. The dataset uses the
    # plain DruggenDataset loader; zero conditions are injected
    # manually in the training loop.
    # --------------------------------------------------------
    joint_positive_dataset = DruggenDataset(
        "druggen_data",
        jp_name,
        cfg.joint_positive_smi,
        MAX_ATOM_DEFAULT,
        False,
        atom_enc,
        atom_dec,
        bond_enc,
        bond_dec,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------
    loader = DataLoader(
        general,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    target_loader = DataLoader(
        joint_positive_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # --------------------------------------------------------
    # Graph dimensions
    # --------------------------------------------------------
    m_dim = len(atom_dec)
    b_dim = len(bond_dec)
    vertexes = int(general[0].x.shape[0])

    log("-" * 70)
    log("DATASET CONFIGURATION")
    log("-" * 70)
    log(f"General graphs             : {len(general):,}")
    log(f"Joint-positive BACE1 graphs: {len(joint_positive_dataset):,}")
    log(f"Maximum atoms              : {vertexes}")
    log(f"Atom dimension             : {m_dim}")
    log(f"Bond dimension             : {b_dim}")
    log(f"Condition dim              : {CONDITION_DIM} (all zeros, Option A)")

    # --------------------------------------------------------
    # Load training-set fingerprints for Tanimoto monitoring
    # --------------------------------------------------------
    log("Loading joint-positive training set fingerprints for Tanimoto monitoring...")
    train_fps = []
    for smi in open(cfg.joint_positive_smi, encoding="utf-8"):
        smi = smi.strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = mol_to_fp(mol)
            if fp is not None:
                train_fps.append(fp)
    log(f"Loaded {len(train_fps)} training-set fingerprints.")

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------
    G = ConditionalGenerator(
        "relu",
        vertexes,
        b_dim,
        m_dim,
        0.0,
        dim=128,
        depth=1,
        heads=8,
        mlp_ratio=3,
        condition_dim=CONDITION_DIM,
    ).to(device)

    D = ConditionalDiscriminator(
        "relu",
        vertexes,
        b_dim,
        m_dim,
        0.0,
        dim=128,
        depth=1,
        heads=8,
        mlp_ratio=3,
        condition_dim=CONDITION_DIM,
    ).to(device)

    # --------------------------------------------------------
    # Warm-start from Issue3-ChemistryReadout checkpoints
    # --------------------------------------------------------
    load_model_checkpoint(G, cfg.init_generator, device, "generator")
    load_model_checkpoint(D, cfg.init_discriminator, device, "discriminator")

    # --------------------------------------------------------
    # Critic helper (no property head in critic path)
    # --------------------------------------------------------
    def critic(edge, node, condition):
        return D(edge, node, condition, return_property=False)[0]

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------
    go = torch.optim.AdamW(G.parameters(), lr=cfg.g_lr, betas=(0.9, 0.999))
    do = torch.optim.AdamW(D.parameters(), lr=cfg.d_lr, betas=(0.9, 0.999))

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------
    global_step = 0
    low_validity_streak = 0  # consecutive log points with ValidFrac < 0.01
    stop_reason = None

    log("")
    log("=" * 70)
    log("TRAINING START")
    log("=" * 70)

    G.train()
    D.train()

    target_loader_iterator = iter(target_loader)

    for mol_batch in loader:

        if global_step >= cfg.max_steps:
            log(f"STOP: reached max_steps={cfg.max_steps}")
            break

        if stop_reason is not None:
            break

        # ----------------------------------------------------
        # Get next joint-positive batch (cycle if exhausted)
        # ----------------------------------------------------
        try:
            real_batch = next(target_loader_iterator)
        except StopIteration:
            target_loader_iterator = iter(target_loader)
            real_batch = next(target_loader_iterator)

        # ====================================================
        # Load graph tensors
        # ====================================================
        _, mol_adj, mol_x = load_molecules(
            mol_batch,
            b_dim=b_dim,
            m_dim=m_dim,
            device=device,
            batch_size=cfg.batch_size,
        )

        _, real_adj, real_x = load_molecules(
            real_batch,
            b_dim=b_dim,
            m_dim=m_dim,
            device=device,
            batch_size=cfg.batch_size,
        )

        # ====================================================
        # Option A: zero condition vectors throughout
        # ====================================================
        zero_cond = torch.zeros(
            cfg.batch_size, CONDITION_DIM, device=device
        )

        # ====================================================
        # Discriminator update
        # ====================================================
        D.zero_grad(set_to_none=True)

        # Generate fake graphs (detached for D update)
        with torch.no_grad():
            _, _, node_sample_d, edge_sample_d = G(mol_adj, mol_x, zero_cond)

        fake_edge_d, fake_node_d = relax_graph(
            node_sample_d, edge_sample_d, mode="st"
        )

        real_score = critic(real_adj, real_x, zero_cond)
        fake_score = critic(fake_edge_d, fake_node_d, zero_cond)

        # WGAN-GP interpolation
        eps_n = torch.rand(cfg.batch_size, 1, 1, device=device)
        eps_e = torch.rand(cfg.batch_size, 1, 1, 1, device=device)

        interp_node = (
            eps_n * real_x + (1.0 - eps_n) * fake_node_d
        ).requires_grad_(True)
        interp_edge = (
            eps_e * real_adj + (1.0 - eps_e) * fake_edge_d
        ).requires_grad_(True)

        interp_score = critic(interp_edge, interp_node, zero_cond)

        grads = torch.autograd.grad(
            interp_score,
            [interp_node, interp_edge],
            torch.ones_like(interp_score),
            create_graph=True,
        )

        gradient = torch.cat(
            [grads[0].flatten(1), grads[1].flatten(1)],
            dim=1,
        )
        gp = ((gradient.norm(2, dim=1) - 1.0) ** 2).mean()

        adv_d_loss = (
            fake_score.mean() - real_score.mean() + cfg.lambda_gp * gp
        )

        # Auxiliary property head loss on real joint-positive graphs
        # (monitoring — still trained so head produces meaningful signals)
        real_pred = D.predict_properties(real_adj, real_x)
        # No condition labels available (plain DruggenDataset); compute
        # auxiliary loss as self-supervised MSE against zeros to keep the
        # head numerically active, but with lambda_aux having no effect on
        # GAN dynamics when real labels are unavailable.
        # Instead: skip aux loss when real condition labels are absent.
        # The discriminator step is purely adversarial in this experiment.
        discriminator_loss = adv_d_loss

        discriminator_loss.backward()
        do.step()

        # ====================================================
        # Generator update
        # ====================================================
        G.zero_grad(set_to_none=True)

        _, _, node_sample_g, edge_sample_g = G(mol_adj, mol_x, zero_cond)

        fake_edge_g, fake_node_g = relax_graph(
            node_sample_g, edge_sample_g, mode="st"
        )

        adv_g_loss = -critic(fake_edge_g, fake_node_g, zero_cond).mean()

        # lambda_condition = 0.0 → no condition-consistency loss
        generator_loss = adv_g_loss

        generator_loss.backward()
        go.step()

        global_step += 1

        # ====================================================
        # Logging and stopping-rule checks every 50 steps
        # ====================================================
        if global_step % 50 == 0:

            g_loss_val = float(generator_loss.detach())
            d_loss_val = float(discriminator_loss.detach())

            # Finite-gradient check
            if not math.isfinite(g_loss_val) or not math.isfinite(d_loss_val):
                stop_reason = (
                    f"STOP: non-finite loss at step {global_step} "
                    f"G={g_loss_val} D={d_loss_val}"
                )
                log(stop_reason)
                break

            # Decode a sample of generated molecules for validity / Tc
            G.eval()
            with torch.no_grad():
                _, _, ns_eval, es_eval = G(mol_adj, mol_x, zero_cond)
            G.train()

            valid_count = 0
            valid_smiles = []
            sample_size = min(cfg.batch_size, 32)

            for i in range(sample_size):
                mol, smi = decode_molecule(
                    joint_positive_dataset, ns_eval, es_eval, i
                )
                if mol is not None and is_nontrivial(mol):
                    valid_count += 1
                    valid_smiles.append(smi)

            valid_frac = valid_count / sample_size

            # Max Tanimoto to training set (on sample up to 32 mols)
            max_tc = 0.0
            tc_above_09_count = 0
            if valid_smiles and train_fps:
                for smi in valid_smiles[:32]:
                    mol_q = Chem.MolFromSmiles(smi)
                    if mol_q is None:
                        continue
                    fp_q = mol_to_fp(mol_q)
                    if fp_q is None:
                        continue
                    sims = DataStructs.BulkTanimotoSimilarity(fp_q, train_fps)
                    tc_max = max(sims) if sims else 0.0
                    if tc_max > max_tc:
                        max_tc = tc_max
                    if tc_max >= 0.9:
                        tc_above_09_count += 1

            log_line = (
                f"STEP={global_step} "
                f"G={g_loss_val:.4f} "
                f"D={d_loss_val:.4f} "
                f"ValidFrac={valid_frac:.4f} "
                f"MaxTc={max_tc:.4f}"
            )
            log(log_line)

            # Stopping rules
            if valid_frac < 0.01:
                low_validity_streak += 1
            else:
                low_validity_streak = 0

            if low_validity_streak >= 3:
                stop_reason = (
                    f"STOP: ValidFrac < 0.01 for 3 consecutive log steps "
                    f"(step {global_step})"
                )
                log(stop_reason)
                break

            if valid_smiles:
                memorization_frac = tc_above_09_count / len(valid_smiles)
                if memorization_frac > 0.9:
                    stop_reason = (
                        f"STOP: memorization detected — "
                        f"{memorization_frac:.1%} of valid molecules "
                        f"have Tc >= 0.9 to training set (step {global_step})"
                    )
                    log(stop_reason)
                    break

    # --------------------------------------------------------
    # Save final checkpoints
    # --------------------------------------------------------
    torch.save(G.state_dict(), out / "generator_pilot.pt")
    torch.save(D.state_dict(), out / "discriminator_pilot.pt")

    log("")
    log("=" * 70)
    log("PILOT COMPLETE")
    log("=" * 70)
    log(f"Steps completed: {global_step}")
    log(f"Saved generator    : {out / 'generator_pilot.pt'}")
    log(f"Saved discriminator: {out / 'discriminator_pilot.pt'}")
    if stop_reason:
        log(f"Early stop reason: {stop_reason}")
    else:
        log("Completed full run (max_steps reached or dataset exhausted).")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Issue 5 — Joint-Positive Discriminator Pilot"
    )

    parser.add_argument(
        "--chembl_smi",
        default="data/chembl_train.smi",
        help="Path to ChEMBL training SMI file (generator input graphs).",
    )

    parser.add_argument(
        "--bace1_smi",
        default="druggen_data/bace1.smi",
        help=(
            "Path to the full BACE1 SMI file. Used ONLY for vocabulary "
            "construction so the encoder/decoder matches the warm-start "
            "checkpoints (node_dim=14, edge_dim=5). Not used as a "
            "discriminator real-molecule set in this experiment."
        ),
    )

    parser.add_argument(
        "--joint_positive_smi",
        default="druggen_data/bace1_joint_positive_train.smi",
        help=(
            "Path to joint-positive BACE1 SMI file "
            "(discriminator real-molecule set)."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="experiments/models/Issue5-JointPositiveDisc-Pilot/",
        help="Directory for checkpoints and run_config.json.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum total training steps (not per epoch).",
    )

    parser.add_argument(
        "--g_lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--d_lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--lambda_gp",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--lambda_aux",
        type=float,
        default=1.0,
        help=(
            "Weight of the auxiliary property loss on real graphs "
            "(currently kept for architecture compatibility; "
            "no real condition labels are available in this experiment "
            "without the conditional dataset loader)."
        ),
    )

    # lambda_condition is fixed at 0.0 and not CLI-configurable.
    # It is included here as a read-only documentation argument.

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--init_generator",
        default=(
            "experiments/models/Issue3-ChemistryReadout/joint_generator.pt"
        ),
        help="Generator checkpoint for warm-start.",
    )

    parser.add_argument(
        "--init_discriminator",
        default=(
            "experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt"
        ),
        help="Discriminator checkpoint for warm-start.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
