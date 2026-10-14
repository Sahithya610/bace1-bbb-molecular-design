"""
DrugGEN training with a differentiable BACE1 surrogate loss.

Why this script exists
----------------------
The baseline training script (train_joint.py) uses a condition-consistency
loss that trains the generator to satisfy D.predict_properties — a 3-layer
MLP reading the discriminator's internal Transformer features.

The independent evaluation oracle is an ExtraTrees model on Morgan ECFP4
fingerprints.  These two representations are structurally decoupled:

    Proof (from condition sensitivity test, issue3_v3 diagnostic):
        requested pIC50 5.5  → external mean pIC50 = 5.454
        requested pIC50 8.0  → external mean pIC50 = 5.444
        Δ = 0.010  (indistinguishable from noise)

The internal head and external oracle produce different predictions for the
same generated molecules (head max ~7.8 vs external max ~6.6 for identical
molecules).

This script adds a differentiable BACE1 surrogate loss: a small MLP
(trained in train_bace1_surrogate.py) that accepts differentiable soft
Morgan fingerprint counts computed directly from the generator's continuous
atom-class and bond-class probability outputs.  The surrogate lives in the
same feature space as the ExtraTrees oracle — Morgan fingerprint bits — so
optimising it pushes the generator toward molecular structures that the
external predictor actually scores highly.

Changes from train_joint.py
----------------------------
1. Loads the pre-trained BACE1 surrogate MLP from
   results/bace1_surrogate_model/bace1_surrogate.pt.
2. Adds soft_morgan_fingerprint(): a differentiable mapping from the
   generator's soft atom/bond probabilities to a 2048-dim bit vector.
3. Adds a generator surrogate loss:
       L_surrogate = MSE(surrogate(soft_fp), target_norm_pIC50)
   weighted by lambda_surrogate.
4. Total generator loss:
       L_G = L_adversarial + λ_condition * L_cond + λ_surrogate * L_surrogate
5. Logs the surrogate's predicted pIC50 alongside the existing head metrics.

The original condition-consistency loss (λ_condition) is retained at its
default value.  Lowering λ_condition is an option if the two objectives
conflict, but start with both active.

Everything else (WGAN-GP, property head, gradient penalty, data pipeline)
is UNCHANGED from train_joint.py.

Differentiable fingerprint approximation
-----------------------------------------
Morgan ECFP2 (radius=1) and ECFP4 (radius=2) bits encode atom-environment
circular substructures.  An exact differentiable implementation would require
message-passing over the graph.  Instead we use a practical soft approximation:

    For each possible (atom_class, bond_class_to_neighbor, neighbor_class)
    triplet, compute the expected count in the batch via the generator's
    soft probabilities, then hash that triplet to a fingerprint bit.

    Concretely:
        fp_soft[b, bit] = sum_{v, e, u: hash(triplet(v,e,u)) == bit}
                              p_node[b,v,:] ⊗ p_edge[b,v,u,:] ⊗ p_node[b,u,:]

    This produces a soft [B, 2048] tensor that is differentiable with
    respect to the generator's node and edge logits.

    The approximation is lower-bound: it covers radius-1 environments only
    (center atom + one bond + neighbor).  The full ECFP4 uses radius 2
    (center + 2 hops).  This means the surrogate's input at training time
    (exact ECFP4) and the loss at generation time (soft ECFP2) are not
    identical.  In practice this is acceptable because:

        1. The surrogate is a continuous approximator of ExtraTrees; small
           input perturbations produce smooth output changes.
        2. The gradient direction (increase predicted pIC50) is still
           correct even if the soft FP is a noisy version of the exact FP.
        3. The pilot will verify empirically whether the external predictor
           score actually shifts upward.

    Soft fp values are clipped to [0, 1] and normalized by atom count so
    the surrogate's input stays in the same range as the binary fingerprints
    it was trained on.

Experiment naming
-----------------
Save to:  experiments/models/Issue3-SurrogateLoss-<tag>/
          (default tag = Pilot100)

Warm-starts from:  experiments/models/Issue3-ChemistryReadout/
                   (the unchanged baseline)

Usage (100-step pilot)
------
    .venv/Scripts/python.exe train_joint_surrogate.py
        --chembl_smi data/chembl_train.smi
        --bace1_smi  druggen_data/bace1.smi
        --labels_csv druggen_data/bace1_continuous_bbb.csv
        --surrogate  results/bace1_surrogate_model/bace1_surrogate.pt
        --init_generator     experiments/models/Issue3-ChemistryReadout/joint_generator.pt
        --init_discriminator experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt
        --output_dir experiments/models/Issue3-SurrogateLoss-Pilot100
        --epochs 1 --max_steps 100
        --lambda_surrogate 0.05
        --batch_size 32
        --seed 1

Note on lambda_surrogate
------------------------
The collapse at lambda_surrogate=1.0 (pilot Experiment D3) showed that the
soft-FP gradient dominates the adversarial signal.  Start at 0.05.  The
train script now also clamps the raw surrogate loss to 2.0 and clips
generator gradient norms to 5.0 as additional guards.

"""

import argparse
import json
import random
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.data.conditional_dataset import ConditionalDruggenDataset
from src.model.conditional_models import (
    ConditionalGenerator,
    ConditionalDiscriminator,
    relax_graph,
)

# ============================================================
# Normalization constants — must match train_joint.py exactly.
# ============================================================

PIC50_MEAN = 6.691315
PIC50_STD = 1.308678
BBB_MEAN = -0.077873
BBB_STD = 0.751623
CONDITION_DIM = 3

# ============================================================
# Fingerprint constants — must match train_bace1_surrogate.py.
# ============================================================

FP_RADIUS = 2      # ECFP4; soft approximation covers radius 1
N_BITS = 2048


# ============================================================
# Surrogate MLP (must match train_bace1_surrogate.py architecture)
# ============================================================

class BACE1SurrogateMLP(nn.Module):
    """
    Frozen (inference-mode) copy of the surrogate trained by
    train_bace1_surrogate.py.  Only the forward pass is needed;
    the weights are loaded from checkpoint and never updated.

    Input: [B, 2048] soft Morgan fingerprint counts, values in [0, 1].
    Output: [B, 1] normalized pIC50.
    """

    def __init__(self, n_bits: int = N_BITS, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Differentiable soft Morgan fingerprint
# ============================================================

def build_fp_hash_table(
    atom_decoder: dict,
    bond_decoder: dict,
    n_bits: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Pre-compute a hash table mapping (center_class, bond_class, neighbor_class)
    triplets to fingerprint bit indices.

    The hash function mimics RDKit's Morgan algorithm radius-1 environment:
        hash = morgan_env_hash(center_atomic_num, bond_type_int, neighbor_atomic_num)
               % n_bits

    We use a deterministic polynomial hash that produces the same bit
    for the same chemical triplet across runs.  The exact hash values
    do NOT need to match RDKit's internal hash — what matters is that
    similar chemical environments map to similar bits and that the
    table is fixed, so gradients are stable.

    Returns
    -------
    hash_table: [m_dim, b_dim, m_dim] int64 tensor of bit indices in [0, n_bits).
    """

    m_dim = len(atom_decoder)   # 14
    b_dim = len(bond_decoder)   # 5

    table = torch.zeros(m_dim, b_dim, m_dim, dtype=torch.long, device=device)

    for ci in range(m_dim):
        center_an = atom_decoder.get(ci, 0)
        for bi in range(b_dim):
            for ni in range(m_dim):
                neighbor_an = atom_decoder.get(ni, 0)
                # Polynomial hash (stable across runs; no Python hash()).
                h = (center_an * 1000003 + bi * 100019 + neighbor_an * 7919)
                table[ci, bi, ni] = h % n_bits

    return table  # [m_dim, b_dim, m_dim]


def soft_morgan_fingerprint(
    node_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    hash_table: torch.Tensor,
    n_bits: int,
    exclude_padding: bool = True,
) -> torch.Tensor:
    """
    Compute a differentiable approximation of the Morgan ECFP4 fingerprint
    from the generator's continuous node and edge logits.

    Parameters
    ----------
    node_logits:  [B, V, m_dim]  — atom class logits (pre-softmax).
    edge_logits:  [B, V, V, b_dim] — bond class logits (pre-softmax).
    hash_table:   [m_dim, b_dim, m_dim]  pre-built bit-index table.
    n_bits:       fingerprint length (2048).
    exclude_padding: zero-out the contribution of atom class 0 (padding).

    Returns
    -------
    fp: [B, n_bits] float tensor, values in [0, 1] after normalization.
        Gradient flows to node_logits and edge_logits.

    Algorithm
    ---------
    1. p_node = softmax(node_logits)   [B, V, m_dim]
    2. p_edge = softmax(edge_logits)   [B, V, V, b_dim]
       (use lower triangle + symmetry mirror from relax_graph — but we
       don't call relax_graph here to preserve full gradient without the
       ST stop-gradient; we work entirely in probability space)
    3. For each bond position (v, u) and each (ci, bi, ni) triplet:
           contribution = p_node[b,v,ci] * p_edge[b,v,u,bi] * p_node[b,u,ni]
       This is the expected probability that position v has atom class ci,
       the bond v→u has bond class bi, and position u has atom class ni.
    4. Accumulate contributions into fp[b, hash_table[ci,bi,ni]] using
       scatter_add.
    5. Normalize by the expected number of non-padding atoms so the
       values are comparable in scale to binary bit vectors (≈ 0 or 1).
    """

    B, V, m_dim = node_logits.shape
    b_dim = edge_logits.shape[-1]
    device = node_logits.device

    p_node = F.softmax(node_logits, dim=-1)          # [B, V, m_dim]
    p_edge = F.softmax(edge_logits, dim=-1)           # [B, V, V, b_dim]

    # Exclude padding class contribution (atom class 0, atomic num 0).
    if exclude_padding:
        p_node_active = p_node.clone()
        p_node_active[:, :, 0] = 0.0
        # Renormalize so remaining classes sum to 1 where any non-padding prob exists.
        p_node_sum = p_node_active.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        p_node_active = p_node_active / p_node_sum
    else:
        p_node_active = p_node

    # Exclude no-bond class (bond class 0) from edge probabilities.
    p_edge_bond = p_edge.clone()
    p_edge_bond[:, :, :, 0] = 0.0
    p_edge_bond_sum = p_edge_bond.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    p_edge_bond = p_edge_bond / p_edge_bond_sum

    # Use only lower triangle (matching relax_graph / matrices2mol).
    mask = torch.tril(torch.ones(V, V, device=device, dtype=p_edge.dtype), diagonal=-1)
    p_edge_bond = p_edge_bond * mask.unsqueeze(0).unsqueeze(-1)  # [B, V, V, b_dim]

    # ----------------------------------------------------------------
    # Compute soft fingerprint via einsum over triplets.
    #
    # For efficiency, iterate over bond classes (only 5) and compute
    # matrix products over (center_class, neighbor_class).
    #
    # fp[b, bit] += sum_{v,u,ci,ni} p_node[b,v,ci] * p_edge[b,v,u,bi]
    #                                * p_node[b,u,ni] * (hash_table[ci,bi,ni] == bit)
    # ----------------------------------------------------------------

    fp = torch.zeros(B, n_bits, device=device, dtype=p_node.dtype)

    for bi in range(b_dim):
        # p_bond_bi: [B, V, V]  — probability that each edge has bond class bi
        p_bond_bi = p_edge_bond[:, :, :, bi]   # [B, V, V]

        # For each (ci, ni) pair, compute:
        #   center_contribution[b, v, ci] = p_node[b,v,ci] * sum_u(p_bond_bi[b,v,u] * p_node_active[b,u,ni])
        #
        # This becomes: center × (p_bond_bi @ p_node_active)
        # p_bond_bi @ p_node_active: [B, V, V] @ [B, V, m_dim] -> [B, V, m_dim]
        # (for each v, weighted sum of neighbor atom probabilities via bond bi)

        neighbor_contrib = torch.bmm(
            p_bond_bi,           # [B, V, V]
            p_node_active,       # [B, V, m_dim]
        )  # [B, V, m_dim]  — for each (b, center_v, neighbor_class)

        # Now outer-product with center class probabilities:
        # triplet_prob[b, v, ci, ni] = p_node_active[b,v,ci] * neighbor_contrib[b,v,ni]
        # Shape: [B, V, m_dim, m_dim]
        triplet = p_node_active.unsqueeze(-1) * neighbor_contrib.unsqueeze(-2)
        # triplet: [B, V, m_dim, m_dim]

        # Flatten over (V, ci, ni):  [B, V*m_dim*m_dim]
        triplet_flat = triplet.reshape(B, -1)  # [B, V * m_dim * m_dim]

        # Bit indices for all (V, ci, ni) positions when bond class = bi.
        # hash_table[:, bi, :] is [m_dim, m_dim] — same for all V positions.
        bits_per_ci_ni = hash_table[:, bi, :]  # [m_dim, m_dim]
        bits_flat = bits_per_ci_ni.reshape(1, -1).expand(V, -1).reshape(-1)
        # bits_flat: [V * m_dim * m_dim]

        # scatter_add over bit dimension.
        fp.scatter_add_(
            1,
            bits_flat.unsqueeze(0).expand(B, -1),
            triplet_flat,
        )

    # ----------------------------------------------------------------
    # Normalize by approximate atom count (expected non-padding atoms).
    # This brings fp values into a comparable range to binary {0, 1}.
    # ----------------------------------------------------------------
    expected_atoms = (1.0 - p_node[:, :, 0]).sum(dim=1, keepdim=True).clamp(min=1.0)
    fp = fp / expected_atoms

    # Clip to [0, 1] to stay in the surrogate's training input range.
    fp = fp.clamp(0.0, 1.0)

    return fp   # [B, n_bits]


# ============================================================
# Helpers (identical to train_joint.py)
# ============================================================

def load_model_checkpoint(model, checkpoint_path, device, model_label):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"{model_label} checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict",
                    "generator_state_dict", "discriminator_state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint format for {model_label}")
    model.load_state_dict(state, strict=True)
    print(f"Loaded {model_label} from: {path}")


def heldout_flags(smiles_list, frac, device):
    if frac <= 0.0:
        return torch.zeros(len(smiles_list), dtype=torch.bool, device=device)
    threshold = int(round(frac * 1000))
    flags = [(zlib.crc32(str(s).encode("utf-8")) % 1000) < threshold
             for s in smiles_list]
    return torch.tensor(flags, dtype=torch.bool, device=device)


def masked_mse(pred, target, keep):
    keep_f = keep.float().unsqueeze(1)
    denom = keep_f.sum().clamp(min=1.0)
    return (((pred - target) ** 2) * keep_f).sum(dim=0) / denom


def r2_per_property(pred, target):
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0).clamp(min=1e-8)
    return 1.0 - ss_res / ss_tot


def pearson_per_property(a, b):
    a = a - a.mean(dim=0)
    b = b - b.mean(dim=0)
    denom = (a.pow(2).sum(dim=0).sqrt() * b.pow(2).sum(dim=0).sqrt()).clamp(min=1e-8)
    return (a * b).sum(dim=0) / denom


# ============================================================
# Training
# ============================================================

def train(cfg):
    if (cfg.init_generator is None) != (cfg.init_discriminator is None):
        raise ValueError("Provide both --init_generator and --init_discriminator, or neither.")

    if cfg.surrogate is None:
        raise ValueError(
            "--surrogate is required. Train the surrogate first:\n"
            "  python train_bace1_surrogate.py --input druggen_data/bace1_continuous_bbb.csv"
        )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("DrugGEN: BACE1 Surrogate Loss (Differentiable Morgan FP Bridge)")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Encoders / decoders
    # --------------------------------------------------------
    atom_enc, atom_dec, bond_enc, bond_dec = get_encoders_decoders(
        cfg.chembl_smi, cfg.bace1_smi, cfg.max_atom
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------
    chembl_name = Path(cfg.chembl_smi).stem + str(cfg.max_atom) + ".pt"
    bace_name   = Path(cfg.bace1_smi).stem  + str(cfg.max_atom) + ".pt"

    general = DruggenDataset(
        cfg.data_dir, chembl_name, cfg.chembl_smi, cfg.max_atom,
        False, atom_enc, atom_dec, bond_enc, bond_dec,
    )

    target = ConditionalDruggenDataset(
        "druggen_data", bace_name, cfg.bace1_smi, cfg.max_atom,
        False, atom_enc, atom_dec, bond_enc, bond_dec,
        labels_csv=cfg.labels_csv,
    )

    loader = DataLoader(general, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    m_dim = len(atom_dec)
    b_dim = len(bond_dec)
    vertexes = int(general[0].x.shape[0])

    print(f"General graphs : {len(general):,}")
    print(f"BACE1 graphs   : {len(target):,}")
    print(f"atom_dim={m_dim}  bond_dim={b_dim}  max_atoms={vertexes}")

    # --------------------------------------------------------
    # Pre-compute hash table for soft fingerprint
    # --------------------------------------------------------
    hash_table = build_fp_hash_table(atom_dec, bond_dec, N_BITS, device)
    print(f"Soft fingerprint hash table: {hash_table.shape} on {device}")

    # --------------------------------------------------------
    # BACE1 surrogate (frozen — only used for generator loss)
    # --------------------------------------------------------
    surrogate = BACE1SurrogateMLP(n_bits=N_BITS, dropout=0.2).to(device)
    surrogate_state = torch.load(cfg.surrogate, map_location=device, weights_only=True)
    surrogate.load_state_dict(surrogate_state, strict=True)
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)
    print(f"Loaded BACE1 surrogate from: {cfg.surrogate}")

    # Sanity-check surrogate on a few training fingerprints.
    # If this fails the surrogate checkpoint is incompatible.
    with torch.no_grad():
        _test_fp = torch.zeros(4, N_BITS, device=device)
        _test_out = surrogate(_test_fp)
        assert _test_out.shape == (4, 1), f"Unexpected surrogate output shape: {_test_out.shape}"
    print("Surrogate forward-pass sanity check passed.")

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------
    G = ConditionalGenerator(
        cfg.act, vertexes, b_dim, m_dim, cfg.dropout,
        cfg.dim, cfg.depth, cfg.heads, cfg.mlp_ratio,
        condition_dim=CONDITION_DIM,
    ).to(device)

    D = ConditionalDiscriminator(
        cfg.act, vertexes, b_dim, m_dim, cfg.ddropout,
        cfg.dim, cfg.ddepth, cfg.heads, cfg.mlp_ratio,
        condition_dim=CONDITION_DIM,
    ).to(device)

    if cfg.init_generator is not None:
        load_model_checkpoint(G, cfg.init_generator, device, "generator")
        load_model_checkpoint(D, cfg.init_discriminator, device, "discriminator")

    def critic(edge, node, condition):
        return D(edge, node, condition, return_property=False)[0]

    # --------------------------------------------------------
    # Optimizers (fresh — no warm-starting optimizer states)
    # --------------------------------------------------------
    go = torch.optim.AdamW(G.parameters(), lr=cfg.g_lr, betas=(cfg.beta1, cfg.beta2))
    do = torch.optim.AdamW(D.parameters(), lr=cfg.d_lr, betas=(cfg.beta1, cfg.beta2))

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = {
        **vars(cfg),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "general_samples": len(general),
        "target_samples": len(target),
        "vertexes": vertexes,
        "node_dim": m_dim,
        "edge_dim": b_dim,
        "condition_dim": CONDITION_DIM,
        "surrogate_loss": {
            "enabled": True,
            "lambda_surrogate": cfg.lambda_surrogate,
            "fp_type": "soft_Morgan_ECFP2_approx",
            "n_bits": N_BITS,
            "note": "soft fingerprint covers radius-1 environments; surrogate trained on ECFP4",
        },
        "condition_consistency_loss": {
            "enabled": True,
            "lambda_condition": cfg.lambda_condition,
            "note": "unchanged from train_joint.py baseline",
        },
        "pIC50_normalization": {"mean": PIC50_MEAN, "std": PIC50_STD},
        "BBB_normalization":   {"mean": BBB_MEAN,   "std": BBB_STD},
        "warm_start": cfg.init_generator is not None,
        "baseline_dir": cfg.init_generator,
        "change_from_baseline": (
            "Added lambda_surrogate * MSE(surrogate(soft_Morgan_fp), target_norm_pIC50) "
            "to generator loss. All other losses, architecture, and data pipeline unchanged."
        ),
    }

    (out / "run_config.json").write_text(json.dumps(info, indent=2))
    print()
    print(json.dumps(info, indent=2))

    # ========================================================
    # Training loop
    # ========================================================
    for epoch in range(1, cfg.epochs + 1):
        G.train()
        D.train()
        surrogate.eval()   # always frozen

        g_loss_total = d_loss_total = 0.0
        aux_total = cond_total = surr_total = 0.0
        aux_prop_total = torch.zeros(2, device=device)
        cond_prop_total = torch.zeros(2, device=device)

        held_pred, held_true = [], []
        fake_read, fake_want = [], []
        surr_pred_log = []   # surrogate predicted pIC50 (de-normalised) for logging

        steps = 0
        target_iter = iter(target_loader)

        for mol_batch in loader:

            # Get next BACE1 batch (cycle).
            try:
                real_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                real_batch = next(target_iter)

            # ------------------------------------------------
            # Prepare tensors
            # ------------------------------------------------
            _, mol_adj, mol_x = load_molecules(
                mol_batch, b_dim=b_dim, m_dim=m_dim, device=device,
                batch_size=cfg.batch_size,
            )
            _, real_adj, real_x = load_molecules(
                real_batch, b_dim=b_dim, m_dim=m_dim, device=device,
                batch_size=cfg.batch_size,
            )

            real_condition = (
                real_batch.condition.to(device).view(cfg.batch_size, CONDITION_DIM)
            )
            desired_condition = real_condition.clone()
            desired_condition[:, 0] = 1.0

            # ================================================
            # Discriminator update  (unchanged from baseline)
            # ================================================
            D.zero_grad(set_to_none=True)

            _, _, node_sample, edge_sample = G(mol_adj, mol_x, desired_condition)

            real_score = critic(real_adj, real_x, real_condition)

            fake_edge_graph, fake_node_graph = relax_graph(node_sample, edge_sample, mode="st")
            fake_score = critic(
                fake_edge_graph.detach(), fake_node_graph.detach(), desired_condition
            )

            # WGAN-GP gradient penalty
            eps_n = torch.rand(cfg.batch_size, 1, 1, device=device)
            eps_e = torch.rand(cfg.batch_size, 1, 1, 1, device=device)

            interp_node = (
                eps_n * real_x + (1.0 - eps_n) * fake_node_graph.detach()
            ).requires_grad_(True)
            interp_edge = (
                eps_e * real_adj + (1.0 - eps_e) * fake_edge_graph.detach()
            ).requires_grad_(True)

            interp_score = critic(interp_edge, interp_node, desired_condition)
            grads = torch.autograd.grad(
                interp_score, [interp_node, interp_edge],
                torch.ones_like(interp_score), create_graph=True,
            )
            gradient = torch.cat([grads[0].flatten(1), grads[1].flatten(1)], dim=1)
            gradient_penalty = ((gradient.norm(2, dim=1) - 1.0) ** 2).mean()

            adversarial_d_loss = (
                fake_score.mean() - real_score.mean() + cfg.lambda_gp * gradient_penalty
            )

            # Auxiliary property loss on real graphs
            real_props = real_condition[:, 1:3]
            real_pred = D.predict_properties(real_adj, real_x)
            holdout = heldout_flags(real_batch.smiles, cfg.aux_holdout_frac, device)
            aux_per_prop = masked_mse(real_pred, real_props, ~holdout)
            aux_loss = aux_per_prop.mean()

            discriminator_loss = adversarial_d_loss + cfg.lambda_aux * aux_loss
            discriminator_loss.backward()
            do.step()

            if holdout.any():
                held_pred.append(real_pred.detach()[holdout])
                held_true.append(real_props.detach()[holdout])

            # ================================================
            # Generator update  (MODIFIED: adds surrogate loss)
            # ================================================
            G.zero_grad(set_to_none=True)

            _, _, node_sample, edge_sample = G(mol_adj, mol_x, desired_condition)

            fake_edge_adv, fake_node_adv = relax_graph(node_sample, edge_sample, mode="st")
            adversarial_g_loss = -critic(
                fake_edge_adv, fake_node_adv, desired_condition
            ).mean()

            # Original condition-consistency loss (unchanged).
            fake_edge_p, fake_node_p = relax_graph(
                node_sample, edge_sample, cfg.property_input
            )
            fake_pred = D.predict_properties(fake_edge_p, fake_node_p)
            wanted_props = desired_condition[:, 1:3]
            cond_per_prop = ((fake_pred - wanted_props) ** 2).mean(dim=0)
            cond_loss = cond_per_prop.mean()

            # --------------------------------------------------
            # NEW: Surrogate loss
            #
            # 1. Compute soft Morgan fingerprint from the generator's
            #    continuous node/edge logits (fully differentiable).
            # 2. Pass through the frozen surrogate MLP to get predicted
            #    normalized pIC50.
            # 3. MSE against the requested (desired) normalized pIC50.
            #
            # This loss directly penalises the Morgan-FP-based pIC50
            # prediction of generated molecules, bridging the gap
            # between the internal property head and the external oracle.
            # --------------------------------------------------
            soft_fp = soft_morgan_fingerprint(
                node_sample,       # raw logits — gradient flows through softmax
                edge_sample,
                hash_table,
                N_BITS,
                exclude_padding=True,
            )   # [B, 2048]

            # Surrogate is frozen — gradients flow through soft_fp to G.
            surr_pred_norm = surrogate(soft_fp)         # [B, 1]
            target_norm_pIC50 = desired_condition[:, 1:2]  # [B, 1]

            surrogate_loss = F.mse_loss(surr_pred_norm, target_norm_pIC50)

            # --------------------------------------------------
            # Surrogate loss stabilization.
            #
            # The soft-FP gradient is consistently large (raw
            # loss hits the clamp ceiling every step at λ=0.05).
            # We use two complementary guards:
            #
            # 1. Clamp the raw loss to 0.5 (in normalized pIC50²
            #    space, this is already generous — it corresponds
            #    to a 0.7 pIC50-unit residual per sample).
            #
            # 2. Scale by the ratio of the adversarial loss
            #    magnitude to the surrogate loss magnitude, so the
            #    surrogate never contributes more than
            #    cfg.lambda_surrogate × |adversarial_loss|.
            #    This is a soft gradient-magnitude normalization
            #    that keeps the surrogate as a directional hint
            #    rather than a dominant signal.
            # --------------------------------------------------
            surrogate_loss_clamped = surrogate_loss.clamp(max=0.5)

            # Relative scaling: keep surrogate ≤ lambda_surrogate
            # times the magnitude of the adversarial signal.
            adv_mag = adversarial_g_loss.detach().abs().clamp(min=0.1)
            surr_mag = surrogate_loss_clamped.detach().clamp(min=1e-6)
            relative_scale = (adv_mag / surr_mag).clamp(max=1.0)

            generator_loss = (
                adversarial_g_loss
                + cfg.lambda_condition * cond_loss
                + cfg.lambda_surrogate * relative_scale * surrogate_loss_clamped
            )
            generator_loss.backward()

            # Gradient norm clipping on the generator.
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)

            go.step()

            # Log surrogate predictions (de-normalized for readability).
            with torch.no_grad():
                surr_pred_log.append(
                    (surr_pred_norm.detach() * PIC50_STD + PIC50_MEAN)
                )

            fake_read.append(fake_pred.detach())
            fake_want.append(wanted_props.detach())

            g_loss_total  += float(generator_loss.detach())
            d_loss_total  += float(discriminator_loss.detach())
            aux_total     += float(aux_loss.detach())
            cond_total    += float(cond_loss.detach())
            surr_total    += float(surrogate_loss_clamped.detach())
            aux_prop_total  += aux_per_prop.detach()
            cond_prop_total += cond_per_prop.detach()

            steps += 1
            if cfg.max_steps is not None and steps >= cfg.max_steps:
                break

        # ====================================================
        # Epoch statistics
        # ====================================================
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"G={g_loss_total/steps:.4f} | D={d_loss_total/steps:.4f} | "
            f"Aux={aux_total/steps:.4f} | Cond={cond_total/steps:.4f} | "
            f"Surr(clamped)={surr_total/steps:.4f} | Steps={steps}"
        )

        aux_mean  = (aux_prop_total  / steps).tolist()
        cond_mean = (cond_prop_total / steps).tolist()
        print(f"    head train MSE : pIC50={aux_mean[0]:.4f}  logBB={aux_mean[1]:.4f}")

        if held_pred:
            hp = torch.cat(held_pred)
            ht = torch.cat(held_true)
            r2 = r2_per_property(hp, ht).tolist()
            print(
                f"    head HELD-OUT R²: pIC50={r2[0]:.3f}  logBB={r2[1]:.3f}  "
                f"(n={hp.shape[0]})"
            )

        fr = torch.cat(fake_read)
        fw = torch.cat(fake_want)
        corr = pearson_per_property(fr, fw).tolist()
        print(f"    G->head cond MSE: pIC50={cond_mean[0]:.4f}  logBB={cond_mean[1]:.4f}")
        print(
            f"    corr(requested, head reading of generated): "
            f"pIC50={corr[0]:.3f}  logBB={corr[1]:.3f}"
        )

        # Surrogate predictions for generated molecules.
        surr_all = torch.cat(surr_pred_log).cpu().numpy().ravel()
        print(
            f"    surrogate pIC50 of generated (mean={surr_all.mean():.4f} "
            f"max={surr_all.max():.4f})  "
            f"[raw pIC50, not external oracle — verify with predict_joint.py]"
        )

        # ====================================================
        # Atom-class diversity check (collapse detector)
        #
        # Run a tiny forward pass (1 batch) with no grad to
        # check the argmax atom-class distribution.  If carbon
        # (class 3, atomic num 6) drops below 10% of non-padding
        # positions, the generator is collapsing.  Abort to
        # preserve the last good checkpoint.
        # ====================================================
        G.eval()
        with torch.no_grad():
            _diag_batch = next(iter(loader))
            _, _da, _dx = load_molecules(
                _diag_batch, b_dim=b_dim, m_dim=m_dim,
                device=device, batch_size=cfg.batch_size,
            )
            _dc = torch.zeros(cfg.batch_size, CONDITION_DIM, device=device)
            _dc[:, 0] = 1.0
            _dc[:, 1] = (8.0 - PIC50_MEAN) / PIC50_STD
            _dc[:, 2] = (0.0 - BBB_MEAN) / BBB_STD
            _, _, _dn, _de = G(_da, _dx, _dc)
            _cls = _dn.argmax(dim=-1).cpu()  # [B, V]

        _counts = torch.bincount(_cls.flatten(), minlength=m_dim).float()
        _non_pad = _counts[1:].sum().clamp(min=1.0)
        _c_frac = _counts[3] / (_non_pad + _counts[0])   # C / total positions
        _h_frac = _counts[1] / (_non_pad + _counts[0])   # H / total positions
        print(
            f"    atom-class check (argmax, high-pIC50 cond): "
            f"C={_c_frac:.3f}  H={_h_frac:.3f}  "
            f"PAD={_counts[0]/(_counts.sum()):.3f}"
        )
        G.train()

        COLLAPSE_C_THRESHOLD = 0.10   # carbon fraction below this = collapse
        COLLAPSE_H_THRESHOLD = 0.80   # hydrogen fraction above this = collapse
        if _c_frac < COLLAPSE_C_THRESHOLD or _h_frac > COLLAPSE_H_THRESHOLD:
            print(
                f"\nCOLLAPSE DETECTED at epoch {epoch}: "
                f"C={_c_frac:.3f} (threshold >{COLLAPSE_C_THRESHOLD}), "
                f"H={_h_frac:.3f} (threshold <{COLLAPSE_H_THRESHOLD}).\n"
                f"Aborting training to preserve checkpoints. "
                f"Reduce --lambda_surrogate and retry."
            )
            # Save what we have, then stop.
            torch.save(G.state_dict(), out / "joint_generator.pt")
            torch.save(D.state_dict(), out / "joint_discriminator.pt")
            break

        # ====================================================
        # Save epoch checkpoints
        # ====================================================
        torch.save(G.state_dict(), out / f"generator_epoch_{epoch:03d}.pt")
        torch.save(D.state_dict(), out / f"discriminator_epoch_{epoch:03d}.pt")

    # Final models
    torch.save(G.state_dict(), out / "joint_generator.pt")
    torch.save(D.state_dict(), out / "joint_discriminator.pt")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Saved generator:     {out / 'joint_generator.pt'}")
    print(f"Saved discriminator: {out / 'joint_discriminator.pt'}")


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description="DrugGEN training with differentiable BACE1 surrogate loss."
)

parser.add_argument("--chembl_smi",   required=True)
parser.add_argument("--bace1_smi",    required=True)
parser.add_argument("--labels_csv",   required=True)
parser.add_argument("--surrogate",    required=True,
                    help="Path to bace1_surrogate.pt from train_bace1_surrogate.py.")
parser.add_argument("--data_dir",     default="data")
parser.add_argument("--output_dir",
                    default="experiments/models/Issue3-SurrogateLoss-Pilot100")
parser.add_argument("--init_generator",    default=None)
parser.add_argument("--init_discriminator", default=None)
parser.add_argument("--max_atom",     type=int,   default=45)
parser.add_argument("--batch_size",   type=int,   default=128)
parser.add_argument("--epochs",       type=int,   default=10)
parser.add_argument("--max_steps",    type=int,   default=None)
parser.add_argument("--dim",          type=int,   default=128)
parser.add_argument("--depth",        type=int,   default=1)
parser.add_argument("--ddepth",       type=int,   default=1)
parser.add_argument("--heads",        type=int,   default=8)
parser.add_argument("--mlp_ratio",    type=int,   default=3)
parser.add_argument("--dropout",      type=float, default=0.0)
parser.add_argument("--ddropout",     type=float, default=0.0)
parser.add_argument("--g_lr",         type=float, default=1e-5)
parser.add_argument("--d_lr",         type=float, default=1e-5)
parser.add_argument("--beta1",        type=float, default=0.9)
parser.add_argument("--beta2",        type=float, default=0.999)
parser.add_argument("--lambda_gp",    type=float, default=10.0)
parser.add_argument("--lambda_aux",   type=float, default=1.0)
parser.add_argument("--lambda_condition", type=float, default=1.0)
parser.add_argument("--lambda_surrogate", type=float, default=0.05,
                    help=(
                        "Weight for the differentiable surrogate loss. "
                        "Default 0.05 (safe starting point after pilot collapse at 1.0). "
                        "The surrogate gradient magnitude is much larger than the "
                        "condition-consistency gradient, so start small. "
                        "Increase to 0.1, 0.2, 0.5 only after confirming no collapse "
                        "(carbon fraction > 10%%, hydrogen fraction < 80%%)."
                    ))
parser.add_argument("--property_input", default="st",
                    choices=["st", "softmax", "logits"])
parser.add_argument("--aux_holdout_frac", type=float, default=0.1)
parser.add_argument("--act",           default="relu",
                    choices=["relu", "leaky", "sigmoid", "tanh"])
parser.add_argument("--seed",          type=int,   default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
