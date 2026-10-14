"""
Issue 5 Recovery — Checkpointed Joint-Positive Discriminator Training.

This script executes a controlled rerun of the joint-positive discriminator
experiment with mandatory intermediate checkpointing every 50 steps:
    50, 100, 150, 200, 250, 300, 350, 400, 450, 500

Key features:
1. Full state checkpoint saving every 50 steps (atomic write + load verification).
2. Standalone generator and discriminator checkpoints saved at each interval.
3. Checkpoint manifest maintained in experiments/models/Issue5-JointPositive-Checkpointed/checkpoint_manifest.json.
4. Comprehensive diagnostics:
   - G/D losses and gradient norms (pre- and post-clipping).
   - Node and edge logit statistics (mean, std, min, max, absmax).
   - Argmax atom-class distributions and padding frequencies.
   - Training-time validity, uniqueness, and Morgan Tanimoto similarity to the
     187 joint-positive training molecules.
   - Diagnostic log written to logs/issue5_checkpointed_pilot_log.txt and logs/issue5_training_diagnostics.jsonl.
5. Early stopping checks:
   - 3 consecutive intervals of ValidFrac < 0.01
   - Non-finite loss or gradient norm
   - Severe atom-class collapse (pad_fraction > 0.95 or single atom class > 0.95)
   - Memorization (>90% valid molecules with Tc >= 0.9 to training set).
6. Preserves the step-300 checkpoint even if subsequent training collapses.
7. Original baseline models, historical directories, and files remain untouched.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import time
import zlib

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

# Constants
CONDITION_DIM = 3
MAX_ATOM_DEFAULT = 45


def load_model_checkpoint(model, checkpoint_path, device, model_label):
    """Load model weights with strict=False to tolerate minor architecture differences."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"{model_label} checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
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
        raise TypeError(f"Unexpected checkpoint format for {model_label}: {type(state).__name__}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded {model_label} from: {path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)}, strict=False)"
    )


def decode_molecule(dataset, node_sample, edge_sample, idx):
    """Decode a single generated graph to an RDKit molecule. Returns (mol, smiles_str) or (None, None)."""
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


def compute_grad_norm(parameters, norm_type=2.0):
    """Compute total L2 gradient norm across a parameter list."""
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.norm(
        torch.stack([torch.norm(g, norm_type) for g in grads]),
        norm_type,
    )
    return float(total_norm.cpu().item())


def atomic_torch_save(obj, target_path):
    """Atomically write a torch checkpoint via temporary file replace."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, target_path)


def save_checkpoint(
    step,
    G,
    D,
    go,
    do,
    cfg,
    run_config,
    out_dir,
    manifest,
    manifest_path,
    is_final=False,
):
    """Save full checkpoint and standalone generator/discriminator weights atomically."""
    out_dir = Path(out_dir)
    suffix = "final" if is_final else f"step_{step}"
    ckpt_path = out_dir / f"checkpoint_{suffix}.pt"
    g_path = out_dir / f"generator_{suffix}.pt"
    d_path = out_dir / f"discriminator_{suffix}.pt"

    checkpoint_data = {
        "step": step,
        "generator_state_dict": G.state_dict(),
        "model_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "generator_optimizer_state_dict": go.state_dict(),
        "discriminator_optimizer_state_dict": do.state_dict(),
        "config": run_config,
        "seed": cfg.seed,
        "scheduler_state_dict": None,
        "is_final": is_final,
    }

    atomic_torch_save(checkpoint_data, ckpt_path)
    atomic_torch_save(G.state_dict(), g_path)
    atomic_torch_save(D.state_dict(), d_path)

    # Load-test immediately to verify integrity
    verify_full = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "generator_state_dict" in verify_full, "Full checkpoint verification failed"
    verify_g = torch.load(g_path, map_location="cpu", weights_only=False)
    assert isinstance(verify_g, dict), "Generator checkpoint verification failed"

    manifest_entry = {
        "step": step,
        "label": suffix,
        "checkpoint_file": str(ckpt_path.resolve()),
        "generator_file": str(g_path.resolve()),
        "discriminator_file": str(d_path.resolve()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verified": True,
        "is_final": is_final,
    }
    manifest["checkpoints"].append(manifest_entry)

    # Save manifest atomically
    tmp_m = manifest_path.with_suffix(".tmp")
    with open(tmp_m, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_m, manifest_path)

    return ckpt_path, g_path, d_path


def train(cfg):
    start_time = time.time()

    # Reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Directories
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "issue5_checkpointed_pilot_log.txt"
    diagnostics_jsonl_path = logs_dir / "issue5_training_diagnostics.jsonl"
    manifest_path = out / "checkpoint_manifest.json"

    # Reset/initialize log files
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")
    with open(diagnostics_jsonl_path, "w", encoding="utf-8") as f:
        f.write("")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    manifest = {
        "experiment": "Issue5-JointPositive-Checkpointed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": vars(cfg),
        "checkpoints": [],
    }

    # Document gradient clipping decision
    clipping_decision = (
        f"Gradient clipping threshold = {cfg.clip_grad_norm}."
        if cfg.clip_grad_norm > 0
        else "Gradient clipping disabled (clip_grad_norm=0.0). "
             "Decision rationale: Previous audit diagnostics established that collapse was mode/validity "
             "collapse without loss divergence or NaN gradients. Pre- and post-clip norms are logged."
    )

    run_config = {
        "experiment": "Issue5-JointPositive-Checkpointed",
        "objective": (
            "Controlled rerun of Issue 5 joint-positive discriminator training with intermediate "
            "checkpoints saved every 50 steps (50, 100, 150, 200, 250, 300, 350, 400, 450, 500) "
            "to preserve pre-collapse model states and evaluate generation with production pipeline."
        ),
        "approach": "Option A: zero condition vector passed to ConditionalGenerator, lambda_condition=0.0",
        "warm_start_generator": cfg.init_generator,
        "warm_start_discriminator": cfg.init_discriminator,
        "chembl_smi": cfg.chembl_smi,
        "bace1_smi_for_vocab": cfg.bace1_smi,
        "joint_positive_train_smi": cfg.joint_positive_smi,
        "batch_size": cfg.batch_size,
        "max_steps": cfg.max_steps,
        "checkpoint_interval": cfg.checkpoint_interval,
        "g_lr": cfg.g_lr,
        "d_lr": cfg.d_lr,
        "lambda_gp": cfg.lambda_gp,
        "lambda_aux": cfg.lambda_aux,
        "lambda_condition": 0.0,
        "clip_grad_norm": cfg.clip_grad_norm,
        "clipping_decision": clipping_decision,
        "seed": cfg.seed,
    }

    (out / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    log("=" * 78)
    log("Issue 5 Recovery — Checkpointed Joint-Positive Discriminator Training")
    log("=" * 78)
    log(json.dumps(run_config, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\nDevice: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    # Vocab and datasets
    atom_enc, atom_dec, bond_enc, bond_dec = get_encoders_decoders(
        cfg.chembl_smi,
        cfg.bace1_smi,
        MAX_ATOM_DEFAULT,
    )

    chembl_name = Path(cfg.chembl_smi).stem + str(MAX_ATOM_DEFAULT) + ".pt"
    jp_name = Path(cfg.joint_positive_smi).stem + str(MAX_ATOM_DEFAULT) + ".pt"

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

    loader = DataLoader(general, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(joint_positive_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    m_dim = len(atom_dec)
    b_dim = len(bond_dec)
    vertexes = int(general[0].x.shape[0])

    log("\n" + "-" * 78)
    log("DATASET CONFIGURATION")
    log("-" * 78)
    log(f"General graphs             : {len(general):,}")
    log(f"Joint-positive BACE1 graphs: {len(joint_positive_dataset):,}")
    log(f"Maximum atoms              : {vertexes}")
    log(f"Atom dimension             : {m_dim}")
    log(f"Bond dimension             : {b_dim}")
    log(f"Condition dim              : {CONDITION_DIM} (Option A: all zeros)")

    # Load 187 joint-positive training fingerprints
    log("\nLoading joint-positive training set fingerprints for Tanimoto monitoring...")
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

    # Initialize models
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

    # Warm-start
    load_model_checkpoint(G, cfg.init_generator, device, "generator")
    load_model_checkpoint(D, cfg.init_discriminator, device, "discriminator")

    def critic(edge, node, condition):
        return D(edge, node, condition, return_property=False)[0]

    # Optimizers
    go = torch.optim.AdamW(G.parameters(), lr=cfg.g_lr, betas=(0.9, 0.999))
    do = torch.optim.AdamW(D.parameters(), lr=cfg.d_lr, betas=(0.9, 0.999))

    global_step = 0
    low_validity_streak = 0
    stop_reason = None

    log("\n" + "=" * 78)
    log("TRAINING START")
    log("=" * 78)

    G.train()
    D.train()
    target_loader_iterator = iter(target_loader)

    for mol_batch in loader:
        if global_step >= cfg.max_steps or stop_reason is not None:
            break

        try:
            real_batch = next(target_loader_iterator)
        except StopIteration:
            target_loader_iterator = iter(target_loader)
            real_batch = next(target_loader_iterator)

        _, mol_adj, mol_x = load_molecules(
            mol_batch, b_dim=b_dim, m_dim=m_dim, device=device, batch_size=cfg.batch_size
        )
        _, real_adj, real_x = load_molecules(
            real_batch, b_dim=b_dim, m_dim=m_dim, device=device, batch_size=cfg.batch_size
        )

        zero_cond = torch.zeros(cfg.batch_size, CONDITION_DIM, device=device)

        # ----------------------------------------------------
        # Discriminator Step
        # ----------------------------------------------------
        D.zero_grad(set_to_none=True)
        with torch.no_grad():
            _, _, node_sample_d, edge_sample_d = G(mol_adj, mol_x, zero_cond)

        fake_edge_d, fake_node_d = relax_graph(node_sample_d, edge_sample_d, mode="st")
        real_score = critic(real_adj, real_x, zero_cond)
        fake_score = critic(fake_edge_d, fake_node_d, zero_cond)

        # WGAN-GP interpolation
        eps_n = torch.rand(cfg.batch_size, 1, 1, device=device)
        eps_e = torch.rand(cfg.batch_size, 1, 1, 1, device=device)
        interp_node = (eps_n * real_x + (1.0 - eps_n) * fake_node_d).requires_grad_(True)
        interp_edge = (eps_e * real_adj + (1.0 - eps_e) * fake_edge_d).requires_grad_(True)
        interp_score = critic(interp_edge, interp_node, zero_cond)

        grads = torch.autograd.grad(
            interp_score,
            [interp_node, interp_edge],
            torch.ones_like(interp_score),
            create_graph=True,
        )
        gradient = torch.cat([grads[0].flatten(1), grads[1].flatten(1)], dim=1)
        gp = ((gradient.norm(2, dim=1) - 1.0) ** 2).mean()

        adv_d_loss = fake_score.mean() - real_score.mean() + cfg.lambda_gp * gp
        discriminator_loss = adv_d_loss
        discriminator_loss.backward()

        d_grad_norm_pre = compute_grad_norm(D.parameters())
        if cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.clip_grad_norm)
            d_grad_norm_post = compute_grad_norm(D.parameters())
        else:
            d_grad_norm_post = d_grad_norm_pre

        do.step()

        # ----------------------------------------------------
        # Generator Step
        # ----------------------------------------------------
        G.zero_grad(set_to_none=True)
        _, _, node_sample_g, edge_sample_g = G(mol_adj, mol_x, zero_cond)
        fake_edge_g, fake_node_g = relax_graph(node_sample_g, edge_sample_g, mode="st")

        adv_g_loss = -critic(fake_edge_g, fake_node_g, zero_cond).mean()
        generator_loss = adv_g_loss
        generator_loss.backward()

        g_grad_norm_pre = compute_grad_norm(G.parameters())
        if cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.clip_grad_norm)
            g_grad_norm_post = compute_grad_norm(G.parameters())
        else:
            g_grad_norm_post = g_grad_norm_pre

        go.step()

        global_step += 1

        # ----------------------------------------------------
        # Logging, diagnostics, and checkpoint saving
        # ----------------------------------------------------
        if global_step % cfg.checkpoint_interval == 0:
            elapsed = time.time() - start_time
            g_loss_val = float(generator_loss.detach().cpu().item())
            d_loss_val = float(discriminator_loss.detach().cpu().item())

            # 1. Non-finite checks
            losses_finite = math.isfinite(g_loss_val) and math.isfinite(d_loss_val)
            grads_finite = (
                math.isfinite(g_grad_norm_pre)
                and math.isfinite(d_grad_norm_pre)
                and math.isfinite(g_grad_norm_post)
                and math.isfinite(d_grad_norm_post)
            )

            # 2. Evaluate sample generation under eval mode
            G.eval()
            with torch.no_grad():
                _, _, ns_eval, es_eval = G(mol_adj, mol_x, zero_cond)
            G.train()

            # Node and edge logit statistics
            node_mean = float(ns_eval.mean().cpu().item())
            node_std = float(ns_eval.std().cpu().item())
            node_min = float(ns_eval.min().cpu().item())
            node_max = float(ns_eval.max().cpu().item())
            node_absmax = float(ns_eval.abs().max().cpu().item())

            edge_mean = float(es_eval.mean().cpu().item())
            edge_std = float(es_eval.std().cpu().item())
            edge_min = float(es_eval.min().cpu().item())
            edge_max = float(es_eval.max().cpu().item())
            edge_absmax = float(es_eval.abs().max().cpu().item())

            # Atom class counts and padding frequency
            node_argmax = ns_eval.argmax(dim=-1).cpu().numpy()
            flat_nodes = node_argmax.flatten()
            class_counts = {int(k): int(v) for k, v in Counter(flat_nodes).items()}
            total_atoms = len(flat_nodes)
            pad_count = class_counts.get(0, 0)
            pad_fraction = pad_count / total_atoms if total_atoms else 0.0
            max_class_fraction = max(class_counts.values()) / total_atoms if total_atoms else 0.0

            # Decode sample molecules (strict=True, nontrivial >3 heavy atoms)
            sample_size = min(cfg.batch_size, 32)
            valid_count = 0
            valid_smiles = []
            for i in range(sample_size):
                mol, smi = decode_molecule(joint_positive_dataset, ns_eval, es_eval, i)
                if mol is not None and is_nontrivial(mol):
                    valid_count += 1
                    valid_smiles.append(smi)

            valid_frac = valid_count / sample_size
            unique_smiles = set(valid_smiles)
            unique_count = len(unique_smiles)
            uniqueness_frac = unique_count / valid_count if valid_count > 0 else 0.0

            # Tanimoto similarity to the 187 joint-positive training set
            max_tc = 0.0
            mean_tc = 0.0
            tc_above_09_count = 0
            all_max_sims = []
            if valid_smiles and train_fps:
                for smi in valid_smiles:
                    mol_q = Chem.MolFromSmiles(smi)
                    if mol_q is None:
                        continue
                    fp_q = mol_to_fp(mol_q)
                    if fp_q is None:
                        continue
                    sims = DataStructs.BulkTanimotoSimilarity(fp_q, train_fps)
                    if sims:
                        local_max = max(sims)
                        all_max_sims.append(local_max)
                        if local_max >= 0.9:
                            tc_above_09_count += 1
                if all_max_sims:
                    max_tc = max(all_max_sims)
                    mean_tc = float(np.mean(all_max_sims))
            frac_tc_above_09 = tc_above_09_count / len(valid_smiles) if valid_smiles else 0.0

            # 3. Save checkpoint atomically
            ckpt_path, g_path, d_path = save_checkpoint(
                global_step,
                G,
                D,
                go,
                do,
                cfg,
                run_config,
                out,
                manifest,
                manifest_path,
                is_final=False,
            )

            # 4. Record structured diagnostic record
            diagnostic_record = {
                "step": global_step,
                "elapsed_sec": round(elapsed, 2),
                "losses": {
                    "G_loss": round(g_loss_val, 4),
                    "D_loss": round(d_loss_val, 4),
                    "finite": losses_finite,
                },
                "gradients": {
                    "G_grad_norm_pre": round(g_grad_norm_pre, 4),
                    "G_grad_norm_post": round(g_grad_norm_post, 4),
                    "D_grad_norm_pre": round(d_grad_norm_pre, 4),
                    "D_grad_norm_post": round(d_grad_norm_post, 4),
                    "finite": grads_finite,
                    "clip_threshold": cfg.clip_grad_norm,
                },
                "logits": {
                    "node": {
                        "mean": round(node_mean, 4),
                        "std": round(node_std, 4),
                        "min": round(node_min, 4),
                        "max": round(node_max, 4),
                        "absmax": round(node_absmax, 4),
                    },
                    "edge": {
                        "mean": round(edge_mean, 4),
                        "std": round(edge_std, 4),
                        "min": round(edge_min, 4),
                        "max": round(edge_max, 4),
                        "absmax": round(edge_absmax, 4),
                    },
                },
                "distributions": {
                    "atom_class_counts": class_counts,
                    "pad_fraction": round(pad_fraction, 4),
                    "max_class_fraction": round(max_class_fraction, 4),
                },
                "generation": {
                    "sample_size": sample_size,
                    "valid_count": valid_count,
                    "valid_fraction": round(valid_frac, 4),
                    "unique_count": unique_count,
                    "uniqueness_fraction": round(uniqueness_frac, 4),
                    "max_tanimoto_to_train": round(max_tc, 4),
                    "mean_tanimoto_to_train": round(mean_tc, 4),
                    "count_tc_ge_09": tc_above_09_count,
                    "frac_tc_ge_09": round(frac_tc_above_09, 4),
                },
                "checkpoints": {
                    "full_checkpoint": str(ckpt_path.resolve()),
                    "generator": str(g_path.resolve()),
                    "discriminator": str(d_path.resolve()),
                },
            }

            with open(diagnostics_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(diagnostic_record) + "\n")

            log_line = (
                f"STEP={global_step:>3} | "
                f"G={g_loss_val:+.4f} D={d_loss_val:+.4f} | "
                f"G_norm={g_grad_norm_post:.3f} D_norm={d_grad_norm_post:.3f} | "
                f"Valid={valid_frac:.3f} ({valid_count}/{sample_size}) | "
                f"Uniq={uniqueness_frac:.3f} | "
                f"MaxTc={max_tc:.3f} | "
                f"PAD={pad_fraction:.2%} | "
                f"Time={elapsed:.1f}s"
            )
            log(log_line)

            # 5. Stopping Rules
            if not losses_finite or not grads_finite:
                stop_reason = (
                    f"STOP: Non-finite loss or gradient detected at step {global_step} "
                    f"(losses_finite={losses_finite}, grads_finite={grads_finite})"
                )
                log(stop_reason)
                break

            if pad_fraction > 0.95 or max_class_fraction > 0.95:
                stop_reason = (
                    f"STOP: Severe atom-class collapse detected at step {global_step} "
                    f"(PAD={pad_fraction:.1%}, max_class={max_class_fraction:.1%})"
                )
                log(stop_reason)
                break

            if valid_smiles and frac_tc_above_09 > 0.90:
                stop_reason = (
                    f"STOP: Training set memorization detected at step {global_step} "
                    f"({frac_tc_above_09:.1%} valid molecules have Tc >= 0.9)"
                )
                log(stop_reason)
                break

            if valid_frac < 0.01:
                low_validity_streak += 1
            else:
                low_validity_streak = 0

            if low_validity_streak >= 3:
                stop_reason = (
                    f"STOP: Generated validity collapsed for 3 consecutive intervals "
                    f"(ValidFrac < 0.01 at step {global_step})"
                )
                log(stop_reason)
                break

    # --------------------------------------------------------
    # Save final checkpoints
    # --------------------------------------------------------
    final_ckpt, final_g, final_d = save_checkpoint(
        global_step,
        G,
        D,
        go,
        do,
        cfg,
        run_config,
        out,
        manifest,
        manifest_path,
        is_final=True,
    )

    total_time = time.time() - start_time
    log("\n" + "=" * 78)
    log("TRAINING COMPLETED OR STOPPED")
    log("=" * 78)
    log(f"Total steps completed: {global_step}")
    log(f"Total elapsed time   : {total_time:.2f}s")
    log(f"Final checkpoint     : {final_ckpt}")
    log(f"Final generator      : {final_g}")
    log(f"Final discriminator  : {final_d}")
    log(f"Checkpoint manifest  : {manifest_path}")
    if stop_reason:
        log(f"Early stop triggered : {stop_reason}")
    else:
        log("Completed planned optimization steps without early stop.")

    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Issue 5 Recovery — Checkpointed Joint-Positive Discriminator Training"
    )
    parser.add_argument("--chembl_smi", default="data/chembl_train.smi")
    parser.add_argument("--bace1_smi", default="druggen_data/bace1.smi")
    parser.add_argument("--joint_positive_smi", default="druggen_data/bace1_joint_positive_train.smi")
    parser.add_argument(
        "--output_dir",
        default="experiments/models/Issue5-JointPositive-Checkpointed/",
        help="Target directory for checkpoints and manifest.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save full checkpoint and compute full diagnostics every N steps.",
    )
    parser.add_argument("--g_lr", type=float, default=1e-5)
    parser.add_argument("--d_lr", type=float, default=1e-5)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--lambda_aux", type=float, default=1.0)
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=0.0,
        help=(
            "Max norm for gradient clipping. Default is 0.0 (disabled) per previous audit "
            "findings which established that collapse was mode collapse rather than gradient divergence. "
            "Set to 1.0 to enable clipping if gradient instability is suspected."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--init_generator",
        default="experiments/models/Issue3-ChemistryReadout/joint_generator.pt",
    )
    parser.add_argument(
        "--init_discriminator",
        default="experiments/models/Issue3-ChemistryReadout/joint_discriminator.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
