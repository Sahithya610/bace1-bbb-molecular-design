"""
DrugGEN REINFORCE training with external oracle reward.

Why this script
---------------
Every approach that used an internal proxy head (ET-distill, surrogate loss,
lambda sweeps, padding penalty) failed because:
  - proxy aligns to GAN latent features, not pharmacophoric features
  - high lambda collapses validity (saturation) or chemistry (single atoms)

This script uses the ACTUAL external oracle (calibrated r=0.901 on 8272
training molecules) as the reward signal via REINFORCE (Williams, 1992).

How it works
------------
Each generator step:
  1. Generate fake graphs (forward pass, keep logits for log-prob computation)
  2. Argmax decode to SMILES in numpy (no grad) using dataset.matrices2mol()
  3. Compute Morgan ECFP4 fingerprint for each valid SMILES
  4. Run sklearn oracle → predicted pIC50 (no grad)
  5. Reward = oracle_pIC50 (baseline-subtracted for variance reduction)
  6. REINFORCE loss = -reward * sum(log_prob of argmax node/bond decisions)
  7. Backprop through the log-prob (which IS differentiable w.r.t. logits)

The oracle never touches the computation graph — gradients flow only through
log P(actions | generator params). This is identical to how REINVENT works
for SMILES-based generators.

Key design choices
------------------
- Baseline: running mean of oracle scores (exponential moving average)
  to reduce variance without bias
- Only valid decoded molecules contribute to REINFORCE loss
  (invalid graphs get reward = 0, same as baseline → zero gradient)
- Mixed loss: WGAN-GP (chemistry quality) + REINFORCE (oracle alignment)
  lambda_oracle controls the balance
- No internal property head loss — the oracle IS the signal

Usage (500-step pilot)
----------------------
    .venv\\Scripts\\python.exe train_joint_reinforce.py
        --chembl_smi data/chembl_train.smi
        --bace1_smi  druggen_data/bace1.smi
        --labels_csv druggen_data/bace1_continuous_bbb.csv
        --init_generator     experiments/models/Issue3-ChemistryReadout/joint_generator.pt
        --init_discriminator experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt
        --output_dir experiments/models/Issue3-Reinforce-Pilot500
        --epochs 1 --max_steps 500 --batch_size 32
        --lambda_oracle 5.0 --seed 1
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.data.conditional_dataset import ConditionalDruggenDataset
from src.model.conditional_models import (
    ConditionalGenerator,
    ConditionalDiscriminator,
    relax_graph,
)

# ── Constants ──────────────────────────────────────────────────────────
PIC50_MEAN    = 6.691315
PIC50_STD     = 1.308678
BBB_MEAN      = -0.077873
BBB_STD       = 0.751623
CONDITION_DIM = 3

ORACLE_PATH   = Path("results/bace1_activity_model/bace1_activity_model.joblib")
FP_RADIUS     = 2
FP_NBITS      = 2048


# ── Helpers ────────────────────────────────────────────────────────────

def load_model_checkpoint(model, path, device, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict",
                  "generator_state_dict", "discriminator_state_dict"):
            if isinstance(ckpt.get(k), dict):
                state = ckpt[k]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [{label}] missing keys: {missing[:3]}")
    print(f"Loaded {label} from: {path}")


def canonicalize(smi):
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


def smiles_to_fp(smi):
    """Morgan ECFP4 → numpy float32 array, or None."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, FP_NBITS)
    return np.array(fp, dtype=np.float32)


def decode_batch_to_smiles(node_logits, edge_logits, dataset):
    """
    Argmax-decode a batch of generator logits to SMILES strings.
    Returns list of (smiles_or_None) length B.
    Pure numpy/RDKit — no gradient.
    """
    B = node_logits.shape[0]
    node_labels = node_logits.detach().cpu().numpy().argmax(-1)   # [B, V]
    edge_labels = edge_logits.detach().cpu().numpy().argmax(-1)   # [B, V, V]

    results = []
    for i in range(B):
        try:
            mol = dataset.matrices2mol(node_labels[i], edge_labels[i],
                                       strict=False)
            if mol is None:
                results.append(None)
                continue
            # Keep largest real fragment
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            real = [f for f in frags
                    if any(a.GetAtomicNum() != 0 for a in f.GetAtoms())]
            if not real:
                results.append(None)
                continue
            best = max(real, key=lambda m: m.GetNumAtoms())
            try:
                Chem.SanitizeMol(best)
                smi = Chem.MolToSmiles(best, canonical=True)
                results.append(smi if smi else None)
            except Exception:
                results.append(None)
        except Exception:
            results.append(None)
    return results


def oracle_score_batch(smiles_list, oracle_model):
    """
    Run oracle on a list of SMILES (may contain None).
    Returns numpy array [B] of predicted pIC50;
    None entries get score 0.0.
    """
    scores = np.zeros(len(smiles_list), dtype=np.float32)
    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        if smi is None:
            continue
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    if fps:
        X = np.vstack(fps)
        preds = oracle_model.predict(X).astype(np.float32)
        for j, idx in enumerate(valid_idx):
            scores[idx] = preds[j]
    return scores


def reinforce_loss(node_logits, edge_logits, rewards_tensor):
    """
    REINFORCE loss = -mean_over_valid( reward_i * log_prob_i )

    log_prob_i = sum of log-softmax values at the argmax positions
                 for all node and bond decisions in molecule i.

    node_logits: [B, V, N]
    edge_logits: [B, V, V, E]
    rewards_tensor: [B] float, baseline-subtracted oracle rewards
    """
    B, V, N = node_logits.shape
    E = edge_logits.shape[-1]

    # Log-softmax over class dimension
    log_p_node = torch.log_softmax(node_logits, dim=-1)  # [B, V, N]
    log_p_edge = torch.log_softmax(edge_logits, dim=-1)  # [B, V, V, E]

    # Actions = argmax (detached — we only use them to index)
    node_actions = node_logits.detach().argmax(-1)        # [B, V]
    edge_actions = edge_logits.detach().argmax(-1)        # [B, V, V]

    # Gather log-prob of chosen actions
    lp_node = log_p_node.gather(                          # [B, V]
        -1, node_actions.unsqueeze(-1)).squeeze(-1)
    lp_edge = log_p_edge.gather(                          # [B, V, V]
        -1, edge_actions.unsqueeze(-1)).squeeze(-1)

    # Sum over positions to get log P(trajectory | params) per molecule
    log_prob = lp_node.sum(dim=1) + lp_edge.sum(dim=(1, 2))  # [B]

    # REINFORCE: -E[R * log P]
    loss = -(rewards_tensor * log_prob).mean()
    return loss


def heldout_flags(smiles_list, frac, device):
    flags = torch.zeros(len(smiles_list), dtype=torch.bool, device=device)
    for i, s in enumerate(smiles_list):
        h = abs(hash(str(s))) % 10000
        if h < frac * 10000:
            flags[i] = True
    return flags


def masked_mse(pred, target, mask):
    if not mask.any():
        return torch.zeros(pred.shape[-1], device=pred.device)
    return ((pred[mask] - target[mask]) ** 2).mean(dim=0)


def pearson_per_property(pred, target):
    out = []
    for j in range(pred.shape[1]):
        p = pred[:, j]; t = target[:, j]
        if p.std() < 1e-6 or t.std() < 1e-6:
            out.append(torch.tensor(0.0))
        else:
            out.append(torch.corrcoef(torch.stack([p, t]))[0, 1])
    return torch.stack(out)


def r2_per_property(pred, target):
    ss_res = ((pred - target) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0)
    return 1 - ss_res / (ss_tot + 1e-8)


# ── Main training function ─────────────────────────────────────────────

def train(cfg):
    # ── Reproducibility ──────────────────────────────────────────────
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("DrugGEN: REINFORCE Oracle Training")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Oracle ───────────────────────────────────────────────────────
    print(f"Loading oracle from: {ORACLE_PATH}")
    oracle = joblib.load(ORACLE_PATH)
    print("Oracle loaded.")

    # ── Encoders / Datasets ──────────────────────────────────────────
    atom_enc, atom_dec, bond_enc, bond_dec = get_encoders_decoders(
        cfg.chembl_smi, cfg.bace1_smi, cfg.max_atom
    )
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

    loader = DataLoader(general, batch_size=cfg.batch_size,
                        shuffle=True, drop_last=True)
    target_loader = DataLoader(target, batch_size=cfg.batch_size,
                               shuffle=True, drop_last=True)

    m_dim    = len(atom_dec)
    b_dim    = len(bond_dec)
    vertexes = int(general[0].x.shape[0])

    print(f"General graphs : {len(general):,}")
    print(f"BACE1 graphs   : {len(target):,}")
    print(f"atom_dim={m_dim}  bond_dim={b_dim}  max_atoms={vertexes}")

    # ── Models ───────────────────────────────────────────────────────
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

    if cfg.init_generator:
        load_model_checkpoint(G, cfg.init_generator, device, "generator")
        load_model_checkpoint(D, cfg.init_discriminator, device, "discriminator")

    def critic(edge, node, condition):
        return D(edge, node, condition, return_property=False)[0]

    # ── Optimisers ───────────────────────────────────────────────────
    go = torch.optim.AdamW(G.parameters(), lr=cfg.g_lr,
                           betas=(cfg.beta1, cfg.beta2))
    do = torch.optim.AdamW(D.parameters(), lr=cfg.d_lr,
                           betas=(cfg.beta1, cfg.beta2))

    # ── Output dir ───────────────────────────────────────────────────
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = {**vars(cfg),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "general_samples": len(general),
            "target_samples":  len(target),
            "vertexes": vertexes, "node_dim": m_dim, "edge_dim": b_dim,
            "condition_dim": CONDITION_DIM,
            "change_from_baseline": (
                "REINFORCE oracle loss. Generator receives reward signal "
                "directly from the calibrated sklearn oracle (r=0.901) via "
                "REINFORCE. No internal proxy head condition loss. "
                "WGAN-GP discriminator preserved for chemistry quality."
            )}
    (out / "run_config.json").write_text(json.dumps(info, indent=2))
    print("\n" + json.dumps(info, indent=2))

    # ── Baseline (EMA of oracle scores) ──────────────────────────────
    baseline = None   # initialised on first valid batch
    ema_alpha = 0.05  # how fast the baseline tracks current rewards

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):
        G.train(); D.train()

        g_total = d_total = 0.0
        rf_total = 0.0
        oracle_scores_all = []
        valid_count = valid_total = 0

        # Diagnostic: corr(requested_pIC50, oracle_pIC50_of_generated)
        requested_all, oracle_all = [], []

        steps = 0
        target_iter = iter(target_loader)

        for mol_batch in loader:
            try:
                real_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                real_batch  = next(target_iter)

            _, mol_adj, mol_x = load_molecules(
                mol_batch, b_dim=b_dim, m_dim=m_dim,
                device=device, batch_size=cfg.batch_size)
            _, real_adj, real_x = load_molecules(
                real_batch, b_dim=b_dim, m_dim=m_dim,
                device=device, batch_size=cfg.batch_size)

            real_condition    = real_batch.condition.to(device).view(cfg.batch_size, CONDITION_DIM)
            desired_condition = real_condition.clone()
            desired_condition[:, 0] = 1.0

            # ── Discriminator update (WGAN-GP, unchanged) ────────────
            D.zero_grad(set_to_none=True)
            _, _, node_sample, edge_sample = G(mol_adj, mol_x, desired_condition)
            real_score = critic(real_adj, real_x, real_condition)
            fake_edge, fake_node = relax_graph(node_sample, edge_sample, mode="st")
            fake_score = critic(fake_edge.detach(), fake_node.detach(), desired_condition)

            eps_n = torch.rand(cfg.batch_size, 1, 1, device=device)
            eps_e = torch.rand(cfg.batch_size, 1, 1, 1, device=device)
            in_node = (eps_n * real_x + (1-eps_n) * fake_node.detach()).requires_grad_(True)
            in_edge = (eps_e * real_adj + (1-eps_e) * fake_edge.detach()).requires_grad_(True)
            gp_score = critic(in_edge, in_node, desired_condition)
            grads = torch.autograd.grad(gp_score, [in_node, in_edge],
                                        torch.ones_like(gp_score), create_graph=True)
            gv = torch.cat([grads[0].flatten(1), grads[1].flatten(1)], dim=1)
            gp = ((gv.norm(2, dim=1) - 1.0) ** 2).mean()
            adv_d = fake_score.mean() - real_score.mean() + cfg.lambda_gp * gp
            adv_d.backward()
            do.step()

            # ── Generator update ─────────────────────────────────────
            G.zero_grad(set_to_none=True)
            _, _, node_sample, edge_sample = G(mol_adj, mol_x, desired_condition)
            fake_edge_adv, fake_node_adv = relax_graph(node_sample, edge_sample, mode="st")
            adv_g = -critic(fake_edge_adv, fake_node_adv, desired_condition).mean()

            # ── CONTRASTIVE REINFORCE ────────────────────────────────
            # Split batch: first half LOW pIC50 (5.5), second half HIGH (8.0)
            # This gives within-batch contrast so the gradient learns
            # condition → chemistry coupling, not just global score maximization.
            half = cfg.batch_size // 2
            low_norm  = (5.5 - PIC50_MEAN) / PIC50_STD
            high_norm = (8.0 - PIC50_MEAN) / PIC50_STD

            cond_contrast = desired_condition.clone()
            cond_contrast[:half, 1] = low_norm
            cond_contrast[half:, 1] = high_norm

            _, _, node_rf, edge_rf = G(mol_adj, mol_x, cond_contrast)

            # Decode to SMILES (no grad)
            smiles_list = decode_batch_to_smiles(node_rf, edge_rf, target)

            # Oracle scores
            scores = oracle_score_batch(smiles_list, oracle)
            n_valid_step = int((scores > 0).sum())
            valid_count  += n_valid_step
            valid_total  += cfg.batch_size

            # Requested pIC50 for each sample
            requested_pic50_np = np.array(
                [5.5] * half + [8.0] * (cfg.batch_size - half),
                dtype=np.float32
            )

            # Soft reward: exp(-|oracle - requested| / τ), range [0,1]
            temperature = 1.0
            cond_rewards = np.exp(
                -np.abs(scores - requested_pic50_np) / temperature
            ).astype(np.float32)
            cond_rewards = cond_rewards * (scores > 0).astype(np.float32)

            # EMA baseline
            valid_mask_rf = torch.tensor(scores > 0, device=device)
            if n_valid_step > 0:
                step_mean_reward = float(cond_rewards[scores > 0].mean())
                if baseline is None:
                    baseline = step_mean_reward
                else:
                    baseline = (1 - ema_alpha) * baseline + ema_alpha * step_mean_reward
            bl = baseline if baseline is not None else 0.5

            rewards = torch.tensor(
                cond_rewards - bl * (scores > 0).astype(np.float32),
                device=device, dtype=torch.float32
            )

            # REINFORCE loss
            if valid_mask_rf.any():
                rf_loss = reinforce_loss(node_rf, edge_rf, rewards)
            else:
                rf_loss = torch.tensor(0.0, device=device, requires_grad=False)

            generator_loss = adv_g + cfg.lambda_oracle * rf_loss
            generator_loss.backward()
            go.step()

            g_total   += float(adv_g.detach())
            d_total   += float(adv_d.detach())
            rf_total  += float(rf_loss.detach() if hasattr(rf_loss, 'detach') else rf_loss)
            oracle_scores_all.extend(scores[scores > 0].tolist())

            # Track requested vs oracle (use contrast batch)
            requested_pic50 = requested_pic50_np.tolist()
            requested_all.extend(requested_pic50)
            oracle_all.extend(scores.tolist())
            steps += 1
            if cfg.max_steps is not None and steps >= cfg.max_steps:
                break

        # ── Epoch stats ──────────────────────────────────────────────
        oracle_arr  = np.array(oracle_scores_all) if oracle_scores_all else np.array([0.0])
        valid_frac  = valid_count / max(valid_total, 1)

        # corr(requested, oracle) — key metric
        req_arr = np.array(requested_all)
        orc_arr = np.array(oracle_all)
        if req_arr.std() > 1e-6 and orc_arr.std() > 1e-6:
            corr = float(np.corrcoef(req_arr, orc_arr)[0, 1])
        else:
            corr = 0.0

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"G={g_total/steps:.4f} | D={d_total/steps:.4f} | "
            f"RF={rf_total/steps:.4f} | "
            f"OracleμpIC50={oracle_arr.mean():.3f} | "
            f"ValidFrac={valid_frac:.3f} | "
            f"Baseline={bl:.3f} | Steps={steps}"
        )
        print(
            f"    corr(requested_pIC50, oracle_pIC50_of_generated): {corr:.3f}"
        )
        print(
            f"    oracle pIC50: min={oracle_arr.min():.2f} "
            f"mean={oracle_arr.mean():.2f} max={oracle_arr.max():.2f}"
        )

        torch.save(G.state_dict(), out / f"generator_epoch_{epoch:03d}.pt")
        torch.save(D.state_dict(), out / f"discriminator_epoch_{epoch:03d}.pt")

    torch.save(G.state_dict(), out / "joint_generator.pt")
    torch.save(D.state_dict(), out / "joint_discriminator.pt")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Saved: {out / 'joint_generator.pt'}")


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chembl_smi",    default="data/chembl_train.smi")
    parser.add_argument("--bace1_smi",     default="druggen_data/bace1.smi")
    parser.add_argument("--labels_csv",    default="druggen_data/bace1_continuous_bbb.csv")
    parser.add_argument("--data_dir",      default="data")
    parser.add_argument("--output_dir",    default="experiments/models/Issue3-Reinforce-Pilot500")
    parser.add_argument("--init_generator",     default="experiments/models/Issue3-ChemistryReadout/joint_generator.pt")
    parser.add_argument("--init_discriminator", default="experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt")
    parser.add_argument("--max_atom",  type=int,   default=45)
    parser.add_argument("--batch_size",type=int,   default=32)
    parser.add_argument("--epochs",    type=int,   default=1)
    parser.add_argument("--max_steps", type=int,   default=500)
    parser.add_argument("--dim",       type=int,   default=128)
    parser.add_argument("--depth",     type=int,   default=1)
    parser.add_argument("--ddepth",    type=int,   default=1)
    parser.add_argument("--heads",     type=int,   default=8)
    parser.add_argument("--mlp_ratio", type=int,   default=3)
    parser.add_argument("--dropout",   type=float, default=0.0)
    parser.add_argument("--ddropout",  type=float, default=0.0)
    parser.add_argument("--g_lr",      type=float, default=1e-5)
    parser.add_argument("--d_lr",      type=float, default=1e-5)
    parser.add_argument("--beta1",     type=float, default=0.9)
    parser.add_argument("--beta2",     type=float, default=0.999)
    parser.add_argument("--lambda_gp",     type=float, default=10.0)
    parser.add_argument("--lambda_oracle", type=float, default=5.0,
                        help="Weight for REINFORCE oracle loss term.")
    parser.add_argument("--act",  default="relu")
    parser.add_argument("--seed", type=int, default=1)
    cfg = parser.parse_args()
    train(cfg)
