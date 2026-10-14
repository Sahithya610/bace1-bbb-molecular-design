"""
DrugGEN training with ExtraTrees-distilled property head supervision.

Why this script exists
----------------------
The baseline (train_joint.py) supervises the auxiliary property head with
raw pIC50 labels from the training CSV.  The external evaluation oracle is
an ExtraTrees model on Morgan ECFP4 fingerprints.  Because these two
predictors operate in structurally different feature spaces (Transformer
graph embeddings vs. Morgan bit-vectors), the property head can be highly
accurate on the training molecules without being aligned with the oracle.
The generator then satisfies the head without moving the oracle score.

    Proof: condition sensitivity test Δ = 0.010 pIC50 units (5.454→5.444)
    across requests pIC50=5.5 vs 8.0 after 10 full training epochs.

The soft Morgan fingerprint approach (train_joint_surrogate.py) was tried
and caused generator collapse in all pilots because the fingerprint gradient
competed with the atom-class softmax distribution.

Fix (this script)
-----------------
Re-supervise the property head with *ExtraTrees predictions* on the
training molecules, precomputed by precompute_et_labels.py.

The head now learns to predict "what ExtraTrees would say about this
molecule" from post-Transformer graph features — bridging the two
representation spaces without any differentiable fingerprint approximation.

The generator's condition-consistency loss (head reading of fake graph ≈
requested pIC50) then trains the generator to produce graph features that
the ET-aligned head scores as high-pIC50 — and since the head was trained
to agree with ExtraTrees, this pressure should propagate to the oracle.

Changes from train_joint.py
----------------------------
ONE change only:

    Auxiliary property loss target:
        BEFORE: real_condition[:, 1]  (normalized raw pIC50 from CSV)
        AFTER:  et_pred_norm_pIC50    (ExtraTrees predicted normalized pIC50)

    The logBB target is unchanged (remains normalized predicted logBB).
    All other losses, architecture, data pipeline, and hyperparameters
    are identical to train_joint.py.

The head is still condition-free.  It still reads only the graph structure.
The condition-consistency loss for the generator is still:
    MSE(head(fake_graph), desired_condition[:, 1:3])

The desired_condition pIC50 slot contains the TRAINING molecule's raw
normalized pIC50 (same as baseline), NOT the ExtraTrees prediction.  This
is intentional: at generation time we request a raw pIC50 value (e.g. 8.0),
and we want the head to respond to that — the head just needs to be aligned
with ExtraTrees so that "head says 8.0" ↔ "ExtraTrees would say ~8.0".

Prerequisite
------------
Run precompute_et_labels.py first to generate:
    druggen_data/bace1_et_predictions.csv

Usage (100-step pilot)
----------------------
    .venv\\Scripts\\python.exe train_joint_et_distill.py
        --chembl_smi data/chembl_train.smi
        --bace1_smi  druggen_data/bace1.smi
        --labels_csv druggen_data/bace1_continuous_bbb.csv
        --et_labels  druggen_data/bace1_et_predictions.csv
        --init_generator     experiments/models/Issue3-ChemistryReadout/joint_generator.pt
        --init_discriminator experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt
        --output_dir experiments/models/Issue3-ETDistill-Pilot100
        --epochs 1 --max_steps 100
        --batch_size 32 --seed 1
"""

import argparse
import json
import random
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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

# ============================================================
# Normalization constants — identical to train_joint.py
# ============================================================

PIC50_MEAN = 6.691315
PIC50_STD  = 1.308678
BBB_MEAN   = -0.077873
BBB_STD    = 0.751623
CONDITION_DIM = 3


# ============================================================
# Helpers — identical to train_joint.py
# ============================================================

def load_model_checkpoint(model, checkpoint_path, device, label):
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
    model.load_state_dict(state, strict=True)
    print(f"Loaded {label} from: {path}")


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


def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


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
    missing = required - set(et_df.columns)
    if missing:
        raise ValueError(
            f"ET predictions CSV is missing columns: {sorted(missing)}\n"
            f"Re-run precompute_et_labels.py."
        )

    # Build smiles → et_pred_norm_pIC50 lookup (canonical SMILES keys).
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

    print("=" * 70)
    print("DrugGEN: ExtraTrees-Distilled Property Head")
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

    print(f"General graphs : {len(general):,}")
    print(f"BACE1 graphs   : {len(target):,}")
    print(f"atom_dim={m_dim}  bond_dim={b_dim}  max_atoms={vertexes}")

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
        "change_from_baseline": (
            "Auxiliary property head supervised by ExtraTrees predicted pIC50 "
            "instead of raw training-CSV pIC50.  LogBB target unchanged.  "
            "All other losses, architecture, and data pipeline identical to "
            "train_joint.py (Issue3-ChemistryReadout baseline)."
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

        g_total = d_total = 0.0
        aux_total = cond_total = 0.0
        aux_prop_total  = torch.zeros(2, device=device)
        cond_prop_total = torch.zeros(2, device=device)

        # Diagnostic buffers
        held_pred, held_true = [], []
        fake_read, fake_want = [], []

        # Track how many BACE1 batch SMILES had an ET prediction available.
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
            # Build ET-supervised auxiliary targets.
            #
            # For each molecule in the real BACE1 batch:
            #   - column 0 of aux_target = ET predicted normalized pIC50
            #     (looked up by canonical SMILES; falls back to the raw
            #      normalized pIC50 if the molecule wasn't in the ET CSV)
            #   - column 1 of aux_target = normalized logBB (unchanged)
            # ------------------------------------------------
            aux_target = real_condition[:, 1:3].clone()   # [B, 2]

            batch_smiles = list(real_batch.smiles)
            for b_idx, smi in enumerate(batch_smiles):
                can = canonicalize(smi)
                if can is not None and can in et_lookup:
                    aux_target[b_idx, 0] = et_lookup[can]
                    et_hit += 1
                else:
                    # Fallback: keep the raw normalized pIC50 (same as baseline).
                    et_miss += 1

            # ================================================
            # Discriminator update  (identical to train_joint.py)
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

            # ------------------------------------------------
            # Auxiliary property loss — ET-supervised target
            # ------------------------------------------------
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
            # Generator update  (identical to train_joint.py)
            # ================================================
            G.zero_grad(set_to_none=True)

            _, _, node_sample, edge_sample = G(mol_adj, mol_x, desired_condition)

            fake_edge_adv, fake_node_adv = relax_graph(node_sample, edge_sample, mode="st")
            adv_g = -critic(fake_edge_adv, fake_node_adv, desired_condition).mean()

            fake_edge_p, fake_node_p = relax_graph(
                node_sample, edge_sample, cfg.property_input
            )
            fake_pred   = D.predict_properties(fake_edge_p, fake_node_p)
            wanted_props = desired_condition[:, 1:3]
            cond_per_prop = ((fake_pred - wanted_props) ** 2).mean(dim=0)
            cond_loss = cond_per_prop.mean()

            generator_loss = adv_g + cfg.lambda_condition * cond_loss
            generator_loss.backward()
            go.step()

            fake_read.append(fake_pred.detach())
            fake_want.append(wanted_props.detach())

            g_total   += float(generator_loss.detach())
            d_total   += float(discriminator_loss.detach())
            aux_total += float(aux_loss.detach())
            cond_total += float(cond_loss.detach())
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
            f"G={g_total/steps:.4f} | D={d_total/steps:.4f} | "
            f"Aux={aux_total/steps:.4f} | Cond={cond_total/steps:.4f} | "
            f"Steps={steps} | ET-hit={et_hit} ET-miss={et_miss}"
        )

        aux_mean  = (aux_prop_total  / steps).tolist()
        cond_mean = (cond_prop_total / steps).tolist()
        print(
            f"    head train MSE : pIC50={aux_mean[0]:.4f}  "
            f"logBB={aux_mean[1]:.4f}  (target = ET pred pIC50)"
        )

        if held_pred:
            hp = torch.cat(held_pred)
            ht = torch.cat(held_true)
            r2 = r2_per_property(hp, ht).tolist()
            print(
                f"    head HELD-OUT R²: pIC50={r2[0]:.3f}  logBB={r2[1]:.3f}  "
                f"(n={hp.shape[0]}, vs ET pred target)"
            )

        fr = torch.cat(fake_read)
        fw = torch.cat(fake_want)
        corr = pearson_per_property(fr, fw).tolist()
        print(
            f"    G->head cond MSE: pIC50={cond_mean[0]:.4f}  logBB={cond_mean[1]:.4f}"
        )
        print(
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
    description="DrugGEN training with ExtraTrees-distilled property head."
)
parser.add_argument("--chembl_smi",   required=True)
parser.add_argument("--bace1_smi",    required=True)
parser.add_argument("--labels_csv",   required=True)
parser.add_argument("--et_labels",    required=True,
                    help="Path to bace1_et_predictions.csv from precompute_et_labels.py.")
parser.add_argument("--data_dir",     default="data")
parser.add_argument("--output_dir",
                    default="experiments/models/Issue3-ETDistill-Pilot100")
parser.add_argument("--init_generator",     default=None)
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
parser.add_argument("--property_input", default="st",
                    choices=["st", "softmax", "logits"])
parser.add_argument("--aux_holdout_frac", type=float, default=0.1)
parser.add_argument("--act",           default="relu",
                    choices=["relu", "leaky", "sigmoid", "tanh"])
parser.add_argument("--seed",          type=int,   default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
