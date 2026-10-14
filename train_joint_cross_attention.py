"""
DrugGEN training with Cross-Attention conditioning (Phase 4 pilot).

Hypothesis
----------
Additive condition embedding injects the condition uniformly across ALL 45
atom positions, causing saturation at high pIC50 requests — the generator
cannot "point" the condition to a specific subgraph region.

Cross-attention allows each atom position to QUERY the condition token and
pull in relevant conditioning information with a learned weight per position.
This should break the saturation and produce a real pIC50 gradient.

Architecture change (ONE change from train_joint_et_distill.py)
---------------------------------------------------------------
After TransformerEncoder output and BEFORE readout_condition concatenation:

    condition_token = Linear(condition_dim, dim)(condition).unsqueeze(1)  [B, 1, dim]
    cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
    node_ca_out, _ = cross_attn(
        query  = node,                       # [B, V, dim]
        key    = condition_token,            # [B, 1, dim]
        value  = condition_token,            # [B, 1, dim]
    )
    node = node + node_ca_out               # residual — keeps existing readout path intact

The existing readout_condition concatenation path is UNCHANGED.

Warm-start
----------
Loaded from Issue3-ChemistryReadout with strict=False.
Keys transferred vs. freshly initialized are logged to training log.
"""

import argparse
import json
import random
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.data.conditional_dataset import ConditionalDruggenDataset
from src.model.conditional_models import (
    ConditionalGenerator,
    ConditionalDiscriminator,
    relax_graph,
)
from src.model.layers import TransformerEncoder

# ============================================================
# Normalization constants — identical to train_joint.py
# ============================================================

PIC50_MEAN = 6.691315
PIC50_STD  = 1.308678
BBB_MEAN   = -0.077873
BBB_STD    = 0.751623
CONDITION_DIM = 3


# ============================================================
# Cross-attention enhanced generator
# (inline subclass — no external files changed)
# ============================================================

class CrossAttentionGenerator(ConditionalGenerator):
    """
    ConditionalGenerator + ONE cross-attention layer on nodes.

    The cross-attention layer is inserted after TransformerEncoder and
    BEFORE the readout_condition concatenation. The existing readout_condition
    pathway is fully preserved.

    New parameters (freshly initialized, not loaded from checkpoint):
        cross_cond_proj  : Linear(condition_dim, dim)
        cross_attn       : MultiheadAttention(dim, heads)
    """

    def __init__(
        self,
        act,
        vertexes,
        edges,
        nodes,
        dropout,
        dim=128,
        depth=1,
        heads=8,
        mlp_ratio=3,
        condition_dim=3,
    ):
        super().__init__(
            act=act,
            vertexes=vertexes,
            edges=edges,
            nodes=nodes,
            dropout=dropout,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            condition_dim=condition_dim,
        )

        # --------------------------------------------------------
        # Cross-attention: condition token projection
        # Linear(condition_dim → dim)
        # --------------------------------------------------------
        self.cross_cond_proj = nn.Linear(condition_dim, dim)

        # --------------------------------------------------------
        # Cross-attention layer
        # query = node features  [B, V, dim]
        # key/value = condition token  [B, 1, dim]
        # --------------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
            dropout=dropout,
        )

    def forward(
        self,
        z_e,
        z_n,
        condition,
    ):
        """
        Forward pass with cross-attention conditioning.

        The cross-attention is applied to node features after the
        TransformerEncoder block and BEFORE readout_condition.
        """

        # ========================================================
        # Original DrugGEN input projections
        # ========================================================

        node = self.node_layers(z_n)
        edge = self.edge_layers(z_e)

        # ========================================================
        # Transformer-level condition injection  (unchanged)
        # ========================================================

        c = self.condition_layers(condition).to(node.dtype)
        node = node + c.unsqueeze(1)
        edge = edge + c.unsqueeze(1).unsqueeze(1)

        # ========================================================
        # Preserve symmetric graph representation  (unchanged)
        # ========================================================

        edge = (edge + edge.permute(0, 2, 1, 3)) / 2

        # ========================================================
        # Transformer  (unchanged)
        # ========================================================

        node, edge = self.TransformerEncoder(node, edge)

        # ========================================================
        # NEW: Cross-attention conditioning
        #
        # Condition token:
        #   Linear(condition_dim, dim) → [B, 1, dim]
        #
        # Each of the V atom positions queries the condition token.
        # The residual connection keeps the existing pathway stable.
        # ========================================================

        condition_token = self.cross_cond_proj(
            condition.to(node.dtype)
        ).unsqueeze(1)  # [B, 1, dim]

        node_ca_out, _ = self.cross_attn(
            query=node,                              # [B, V, dim]
            key=condition_token,                     # [B, 1, dim]
            value=condition_token,                   # [B, 1, dim]
        )
        node = node + node_ca_out                   # residual

        # ========================================================
        # ISSUE-3 CHEMISTRY-LEVEL CONDITION-AWARE READOUT  (unchanged)
        # ========================================================

        readout_condition = self.readout_condition(
            condition
        ).to(node.dtype)

        node_condition = (
            readout_condition
            .unsqueeze(1)
            .expand(-1, node.shape[1], -1)
        )

        edge_condition = (
            readout_condition
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, edge.shape[1], edge.shape[2], -1)
        )

        node_readout = torch.cat([node, node_condition], dim=-1)
        edge_readout = torch.cat([edge, edge_condition], dim=-1)

        # ========================================================
        # Final categorical decisions  (unchanged)
        # ========================================================

        node_sample = self.readout_n(node_readout)
        edge_sample = self.readout_e(edge_readout)

        return node, edge, node_sample, edge_sample


# ============================================================
# Helpers — identical to train_joint_et_distill.py
# ============================================================

def load_model_checkpoint_strict_false(model, checkpoint_path, device, label):
    """
    Load checkpoint with strict=False (new cross-attn keys will be init'd fresh).
    Returns (transferred_keys, missing_keys, unexpected_keys).
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict",
                  "generator_state_dict", "discriminator_state_dict"):
            candidate = ckpt.get(k)
            if isinstance(candidate, dict):
                state = candidate
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint format for {label}")
    result = model.load_state_dict(state, strict=False)
    missing  = result.missing_keys
    unexpected = result.unexpected_keys
    # "transferred" = keys that ARE in state and in model (not missing/unexpected)
    state_keys  = set(state.keys())
    model_keys  = set(k for k, _ in model.named_parameters())
    model_keys |= set(k for k, _ in model.named_buffers())
    transferred = state_keys & model_keys
    print(f"Loaded {label} from: {path}  (strict=False)")
    print(f"  Transferred  : {len(transferred)} keys")
    print(f"  Missing keys : {len(missing)} — will be initialized fresh")
    print(f"  Unexpected   : {len(unexpected)} — ignored")
    if missing:
        print("  Missing key list:")
        for k in missing:
            print(f"    {k}")
    return transferred, missing, unexpected


def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


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
        raise ValueError(
            "Provide both --init_generator and --init_discriminator, or neither."
        )

    # --------------------------------------------------------
    # Load ExtraTrees predictions lookup
    # --------------------------------------------------------
    et_path = Path(cfg.et_labels)
    if not et_path.exists():
        raise FileNotFoundError(
            f"ET predictions CSV not found: {et_path}\n"
            "Run precompute_et_labels.py first."
        )

    et_df = pd.read_csv(et_path)
    required = {"canonical_smiles", "et_pred_norm_pIC50"}
    missing_cols = required - set(et_df.columns)
    if missing_cols:
        raise ValueError(
            f"ET predictions CSV is missing columns: {sorted(missing_cols)}\n"
            f"Re-run precompute_et_labels.py."
        )

    et_lookup = {}
    for _, row in et_df.iterrows():
        can = canonicalize(row["canonical_smiles"])
        if can is not None:
            et_lookup[can] = float(row["et_pred_norm_pIC50"])

    print(f"ET prediction lookup loaded: {len(et_lookup):,} molecules")

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

    # --------------------------------------------------------
    # Log file setup
    # --------------------------------------------------------
    log_path = Path(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", buffering=1)

    def log(msg):
        print(msg)
        log_fh.write(msg + "\n")

    log("=" * 70)
    log("DrugGEN: Cross-Attention Conditioning Pilot (Phase 4)")
    log("=" * 70)
    log(f"Device: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Encoders / decoders
    # --------------------------------------------------------
    atom_enc, atom_dec, bond_enc, bond_dec = get_encoders_decoders(
        cfg.chembl_smi, cfg.bace1_smi, cfg.max_atom
    )

    chembl_name = Path(cfg.chembl_smi).stem + str(cfg.max_atom) + ".pt"
    bace_name   = Path(cfg.bace1_smi).stem  + str(cfg.max_atom) + ".pt"

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------
    general = DruggenDataset(
        cfg.data_dir, chembl_name, cfg.chembl_smi, cfg.max_atom,
        False, atom_enc, atom_dec, bond_enc, bond_dec,
    )

    target = ConditionalDruggenDataset(
        "druggen_data", bace_name, cfg.bace1_smi, cfg.max_atom,
        False, atom_enc, atom_dec, bond_enc, bond_dec,
        labels_csv=cfg.labels_csv,
    )

    loader = DataLoader(
        general, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    target_loader = DataLoader(
        target, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )

    m_dim    = len(atom_dec)
    b_dim    = len(bond_dec)
    vertexes = int(general[0].x.shape[0])

    log(f"General graphs : {len(general):,}")
    log(f"BACE1 graphs   : {len(target):,}")
    log(f"atom_dim={m_dim}  bond_dim={b_dim}  max_atoms={vertexes}")

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------
    G = CrossAttentionGenerator(
        cfg.act, vertexes, b_dim, m_dim, cfg.dropout,
        cfg.dim, cfg.depth, cfg.heads, cfg.mlp_ratio,
        condition_dim=CONDITION_DIM,
    ).to(device)

    D = ConditionalDiscriminator(
        cfg.act, vertexes, b_dim, m_dim, cfg.ddropout,
        cfg.dim, cfg.ddepth, cfg.heads, cfg.mlp_ratio,
        condition_dim=CONDITION_DIM,
    ).to(device)

    # --------------------------------------------------------
    # Warm-start with strict=False — log key transfer
    # --------------------------------------------------------
    if cfg.init_generator is not None:
        log("\n--- Generator warm-start ---")
        g_transferred, g_missing, g_unexpected = load_model_checkpoint_strict_false(
            G, cfg.init_generator, device, "generator"
        )
        for k in g_missing:
            log(f"  [FRESH INIT] {k}")
        for k in g_unexpected:
            log(f"  [IGNORED]   {k}")
        log(f"  Generator: {len(g_transferred)} transferred, "
            f"{len(g_missing)} fresh, {len(g_unexpected)} ignored")

        log("\n--- Discriminator warm-start ---")
        d_transferred, d_missing, d_unexpected = load_model_checkpoint_strict_false(
            D, cfg.init_discriminator, device, "discriminator"
        )
        log(f"  Discriminator: {len(d_transferred)} transferred, "
            f"{len(d_missing)} fresh, {len(d_unexpected)} ignored")

    def critic(edge, node, condition):
        return D(edge, node, condition, return_property=False)[0]

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------
    go = torch.optim.AdamW(
        G.parameters(), lr=cfg.g_lr, betas=(cfg.beta1, cfg.beta2)
    )
    do = torch.optim.AdamW(
        D.parameters(), lr=cfg.d_lr, betas=(cfg.beta1, cfg.beta2)
    )

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
        "et_lookup_size": len(et_lookup),
        "pIC50_normalization": {"mean": PIC50_MEAN, "std": PIC50_STD},
        "BBB_normalization":   {"mean": BBB_MEAN,   "std": BBB_STD},
        "architecture_change": (
            "Added ONE cross-attention layer to ConditionalGenerator: "
            "after TransformerEncoder and BEFORE readout_condition concatenation. "
            "Each atom position (query=[B,V,dim]) attends to a condition token "
            "(key=value=Linear(condition_dim,dim).unsqueeze(1) → [B,1,dim]). "
            "Residual: node = node + node_ca_out. "
            "The existing readout_condition path is unchanged. "
            "Warm-started from Issue3-ChemistryReadout with strict=False. "
            "New params (cross_cond_proj, cross_attn) initialized fresh."
        ),
    }

    (out / "run_config.json").write_text(json.dumps(info, indent=2))
    log("\n" + json.dumps(info, indent=2))

    # ========================================================
    # Training loop
    # ========================================================
    for epoch in range(1, cfg.epochs + 1):
        G.train()
        D.train()

        g_total = d_total = 0.0
        aux_total = cond_total = 0.0
        aux_prop_total  = torch.zeros(2, device=device)
        cond_prop_total = torch.zeros(2, device=device)

        held_pred, held_true = [], []
        fake_read, fake_want = [], []
        et_hit = et_miss = 0

        steps = 0
        target_iter = iter(target_loader)

        for mol_batch in loader:

            try:
                real_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                real_batch = next(target_iter)

            # ------------------------------------------------
            # Prepare tensors
            # ------------------------------------------------
            _, mol_adj, mol_x = load_molecules(
                mol_batch, b_dim=b_dim, m_dim=m_dim,
                device=device, batch_size=cfg.batch_size,
            )
            _, real_adj, real_x = load_molecules(
                real_batch, b_dim=b_dim, m_dim=m_dim,
                device=device, batch_size=cfg.batch_size,
            )

            real_condition = (
                real_batch.condition.to(device)
                .view(cfg.batch_size, CONDITION_DIM)
            )
            desired_condition = real_condition.clone()
            desired_condition[:, 0] = 1.0

            # ------------------------------------------------
            # Build ET-supervised auxiliary targets
            # ------------------------------------------------
            aux_target = real_condition[:, 1:3].clone()  # [B, 2]

            batch_smiles = list(real_batch.smiles)
            for b_idx, smi in enumerate(batch_smiles):
                can = canonicalize(smi)
                if can is not None and can in et_lookup:
                    aux_target[b_idx, 0] = et_lookup[can]
                    et_hit += 1
                else:
                    et_miss += 1

            # ================================================
            # Discriminator update
            # ================================================
            D.zero_grad(set_to_none=True)

            _, _, node_sample, edge_sample = G(mol_adj, mol_x, desired_condition)

            real_score = critic(real_adj, real_x, real_condition)

            fake_edge, fake_node = relax_graph(node_sample, edge_sample, mode="st")
            fake_score = critic(
                fake_edge.detach(), fake_node.detach(), desired_condition
            )

            # WGAN-GP
            eps_n = torch.rand(cfg.batch_size, 1, 1, device=device)
            eps_e = torch.rand(cfg.batch_size, 1, 1, 1, device=device)
            interp_node = (eps_n * real_x + (1 - eps_n) * fake_node.detach()).requires_grad_(True)
            interp_edge = (eps_e * real_adj + (1 - eps_e) * fake_edge.detach()).requires_grad_(True)

            interp_score = critic(interp_edge, interp_node, desired_condition)
            grads = torch.autograd.grad(
                interp_score, [interp_node, interp_edge],
                torch.ones_like(interp_score), create_graph=True,
            )
            grad_vec = torch.cat([grads[0].flatten(1), grads[1].flatten(1)], dim=1)
            gp = ((grad_vec.norm(2, dim=1) - 1.0) ** 2).mean()

            adv_d = fake_score.mean() - real_score.mean() + cfg.lambda_gp * gp

            real_pred = D.predict_properties(real_adj, real_x)
            holdout   = heldout_flags(real_batch.smiles, cfg.aux_holdout_frac, device)

            aux_per_prop = masked_mse(real_pred, aux_target, ~holdout)
            aux_loss = aux_per_prop.mean()

            discriminator_loss = adv_d + cfg.lambda_aux * aux_loss
            discriminator_loss.backward()
            do.step()

            if holdout.any():
                held_pred.append(real_pred.detach()[holdout])
                held_true.append(aux_target.detach()[holdout])

            # ================================================
            # Generator update
            # ================================================
            G.zero_grad(set_to_none=True)

            _, _, node_sample, edge_sample = G(mol_adj, mol_x, desired_condition)

            fake_edge_adv, fake_node_adv = relax_graph(node_sample, edge_sample, mode="st")
            adv_g = -critic(fake_edge_adv, fake_node_adv, desired_condition).mean()

            fake_edge_p, fake_node_p = relax_graph(
                node_sample, edge_sample, cfg.property_input
            )
            fake_pred    = D.predict_properties(fake_edge_p, fake_node_p)
            wanted_props = desired_condition[:, 1:3]
            cond_per_prop = ((fake_pred - wanted_props) ** 2).mean(dim=0)
            cond_loss = cond_per_prop.mean()

            generator_loss = adv_g + cfg.lambda_condition * cond_loss
            generator_loss.backward()
            go.step()

            fake_read.append(fake_pred.detach())
            fake_want.append(wanted_props.detach())

            g_total    += float(generator_loss.detach())
            d_total    += float(discriminator_loss.detach())
            aux_total  += float(aux_loss.detach())
            cond_total += float(cond_loss.detach())
            aux_prop_total  += aux_per_prop.detach()
            cond_prop_total += cond_per_prop.detach()

            steps += 1

            # ------------------------------------------------
            # Mid-run collapse check (every 50 steps)
            # ------------------------------------------------
            if steps % 50 == 0:
                log(
                    f"  Step {steps:>4} | G={g_total/steps:.4f} | "
                    f"D={d_total/steps:.4f} | "
                    f"Aux={aux_total/steps:.4f} | "
                    f"Cond={cond_total/steps:.4f}"
                )

            if cfg.max_steps is not None and steps >= cfg.max_steps:
                break

        # ====================================================
        # Epoch statistics
        # ====================================================
        log(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"G={g_total/steps:.4f} | D={d_total/steps:.4f} | "
            f"Aux={aux_total/steps:.4f} | Cond={cond_total/steps:.4f} | "
            f"Steps={steps} | ET-hit={et_hit} ET-miss={et_miss}"
        )

        aux_mean  = (aux_prop_total  / steps).tolist()
        cond_mean = (cond_prop_total / steps).tolist()
        log(
            f"    head train MSE : pIC50={aux_mean[0]:.4f}  "
            f"logBB={aux_mean[1]:.4f}"
        )

        if held_pred:
            hp = torch.cat(held_pred)
            ht = torch.cat(held_true)
            r2 = r2_per_property(hp, ht).tolist()
            log(
                f"    head HELD-OUT R²: pIC50={r2[0]:.3f}  logBB={r2[1]:.3f}  "
                f"(n={hp.shape[0]})"
            )

        fr = torch.cat(fake_read)
        fw = torch.cat(fake_want)
        corr = pearson_per_property(fr, fw).tolist()
        log(
            f"    G->head cond MSE: pIC50={cond_mean[0]:.4f}  logBB={cond_mean[1]:.4f}"
        )
        log(
            f"    corr(requested, head-reading-of-generated): "
            f"pIC50={corr[0]:.3f}  logBB={corr[1]:.3f}"
        )

        # ====================================================
        # Save epoch checkpoints
        # ====================================================
        torch.save(G.state_dict(), out / f"generator_epoch_{epoch:03d}.pt")
        torch.save(D.state_dict(), out / f"discriminator_epoch_{epoch:03d}.pt")

    # Final
    torch.save(G.state_dict(), out / "joint_generator.pt")
    torch.save(D.state_dict(), out / "joint_discriminator.pt")

    log("")
    log("=" * 70)
    log("TRAINING COMPLETE")
    log("=" * 70)
    log(f"Saved generator:     {out / 'joint_generator.pt'}")
    log(f"Saved discriminator: {out / 'joint_discriminator.pt'}")
    log_fh.close()


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description="DrugGEN training with Cross-Attention conditioning (Phase 4)."
)
parser.add_argument("--chembl_smi",   required=True)
parser.add_argument("--bace1_smi",    required=True)
parser.add_argument("--labels_csv",   required=True)
parser.add_argument("--et_labels",    required=True,
                    help="Path to bace1_et_predictions.csv from precompute_et_labels.py.")
parser.add_argument("--data_dir",     default="data")
parser.add_argument("--output_dir",
                    default="experiments/models/Issue4-CrossAttention-Pilot500")
parser.add_argument("--log_file",
                    default="results/cross_attention_pilot/training_log.txt")
parser.add_argument("--init_generator",     default=None)
parser.add_argument("--init_discriminator", default=None)
parser.add_argument("--max_atom",     type=int,   default=45)
parser.add_argument("--batch_size",   type=int,   default=32)
parser.add_argument("--epochs",       type=int,   default=1)
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
parser.add_argument("--property_input", default="st",
                    choices=["st", "softmax", "logits"])
parser.add_argument("--aux_holdout_frac", type=float, default=0.1)
parser.add_argument("--act",           default="relu",
                    choices=["relu", "leaky", "sigmoid", "tanh"])
parser.add_argument("--seed",          type=int,   default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
