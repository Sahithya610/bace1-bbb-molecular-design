"""
Train explicitly conditioned DrugGEN for BACE1 activity + BBB permeability.

Architecture:
    Original DrugGEN Graph-Transformer GAN
    +
    continuous conditioning pathway

Condition:
    [BACE1_target, normalized_pIC50, normalized_logBB]

BACE1_target:
    1.0

pIC50:
    Continuous BACE1 activity value derived from ChEMBL.

    Normalization:
        mean = 6.691315
        std  = 1.308678

BBB:
    Continuous predicted logBB from the B3DB-trained BBB
    regression model.

    Normalization:
        mean = -0.077873
        std  = 0.751623

Training datasets:
    General ChEMBL:
        1,418,299 processed molecular graphs

    BACE1:
        7,682 processed molecular graphs

Sampling:
    Standard shuffled sampling.

    No affinity-enriched sampling is used.
    All BACE1 graphs remain equally eligible.

Objective:
    Generate molecular graphs conditioned simultaneously on:

        1. BACE1 target
        2. desired BACE1 activity
        3. continuous BBB permeability

No binary BBB condition is used.

Experiment:
    Explicit continuous pIC50 + continuous logBB conditioning.

Training-loop correction:
    The BACE1 target loader is cycled while the complete
    general ChEMBL loader is traversed.

    This avoids zip(loader, target_loader), which would stop
    an epoch after the much smaller BACE1 dataset is exhausted.

Smoke-test support:
    --max_steps can limit the number of training steps per epoch.
    If omitted, the complete general ChEMBL dataset is traversed.

Issue 3 (graph-aware auxiliary property head):
    Discriminator
        D.predict_properties(edge, node) -> [pIC50_norm, logBB_norm]
        - uses the POST-Transformer node AND edge representation
          (molecular connectivity), unlike the Issue 2 head which read
          pre-Transformer node features only
        - is CONDITION-FREE (no condition embedding injected), so it
          cannot simply read the requested property back from the
          condition vector
        - is supervised on REAL BACE1 graphs only (labels = the
          condition values of those graphs)
    Generator
        condition-consistency loss: the head's reading of the generated
        graph must match the requested [pIC50_norm, logBB_norm].
        The generated graph is shown to the head in the same discrete
        form that matrices2mol() decodes (see relax_graph).

    Hold-out for diagnostics:
        a deterministic ~10 % of BACE1 graphs (crc32 of the SMILES) are
        excluded from the AUXILIARY loss only. They are still used as
        real samples for the WGAN critic, so the GAN training data is
        unchanged. The held-out R2 of the head shows whether the head
        generalises to unseen molecules (it is the quantity the generator
        is able to exploit).
"""

import argparse
import json
import random
import zlib
from pathlib import Path

import numpy as np
import torch
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
# Normalization constants
# ============================================================

PIC50_MEAN = 6.691315
PIC50_STD = 1.308678

BBB_MEAN = -0.077873
BBB_STD = 0.751623

CONDITION_DIM = 3


def load_model_checkpoint(
    model,
    checkpoint_path,
    device,
    model_label,
    exclude_prefixes=(),
):
    """Load compatible weights, optionally leaving selected layers initialized fresh."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"{model_label} checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    # Support either a raw state_dict or common wrapped checkpoint formats.
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
            f"Unexpected checkpoint format for {model_label}: {type(state).__name__}"
        )

    if not exclude_prefixes:
        model.load_state_dict(state, strict=True)
        print(f"Loaded {model_label} initialization from: {path}")
        return

    # Preserve the shared graph trunk and auxiliary property head while resetting
    # the adversarial Wasserstein scoring head (node_mlp.*) for the corrected
    # critic-input representation. Excluded layers retain fresh constructor weights.
    model_keys = set(model.state_dict().keys())
    filtered_state = {
        key: value
        for key, value in state.items()
        if not any(key.startswith(prefix) for prefix in exclude_prefixes)
    }
    skipped_keys = {
        key for key in model_keys
        if any(key.startswith(prefix) for prefix in exclude_prefixes)
    }

    if not skipped_keys:
        raise RuntimeError(
            f"No model keys matched the requested exclusions: {exclude_prefixes}"
        )

    incompatible = model.load_state_dict(filtered_state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != skipped_keys or unexpected:
        raise RuntimeError(
            "Partial checkpoint loading mismatch. "
            f"Expected missing keys {sorted(skipped_keys)}, "
            f"got missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    print(f"Loaded {model_label} initialization from: {path}")
    print(
        f"Freshly initialized {model_label} layers for prefixes "
        f"{exclude_prefixes}: {len(skipped_keys)} tensors"
    )


# ============================================================
# Auxiliary-loss helpers
# ============================================================

def heldout_flags(smiles_list, frac, device):
    """
    Deterministic hold-out flags (True = excluded from the auxiliary loss).

    Uses crc32 (NOT Python hash(), which is salted per process) so that
    the same molecules are held out in every run and every epoch.
    """

    if frac <= 0.0:
        return torch.zeros(
            len(smiles_list), dtype=torch.bool, device=device
        )

    threshold = int(round(frac * 1000))

    flags = [
        (zlib.crc32(str(s).encode("utf-8")) % 1000) < threshold
        for s in smiles_list
    ]

    return torch.tensor(flags, dtype=torch.bool, device=device)


def masked_mse(pred, target, keep):
    """
    Per-property MSE over the rows where keep == True.

    pred, target: [B, 2]   keep: [B] bool   returns: [2]
    """

    keep_f = keep.float().unsqueeze(1)
    denom = keep_f.sum().clamp(min=1.0)

    return (((pred - target) ** 2) * keep_f).sum(dim=0) / denom


def r2_per_property(pred, target):
    """R2 for each column. pred, target: [N, 2]."""

    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = (
        (target - target.mean(dim=0)) ** 2
    ).sum(dim=0).clamp(min=1e-8)

    return 1.0 - ss_res / ss_tot


def pearson_per_property(a, b):
    """Pearson correlation for each column. a, b: [N, 2]."""

    a = a - a.mean(dim=0)
    b = b - b.mean(dim=0)

    denom = (
        a.pow(2).sum(dim=0).sqrt() * b.pow(2).sum(dim=0).sqrt()
    ).clamp(min=1e-8)

    return (a * b).sum(dim=0) / denom


def train(cfg):

    # Warm-start the generator and discriminator together or not at all.
    if (cfg.init_generator is None) != (cfg.init_discriminator is None):
        raise ValueError(
            "Provide both --init_generator and --init_discriminator, or neither."
        )

    if cfg.reset_critic_head and cfg.init_discriminator is None:
        raise ValueError(
            "--reset_critic_head requires --init_generator and --init_discriminator."
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

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("DrugGEN: Explicit BACE1 Affinity + Continuous BBB Conditioning")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Encoders / decoders
    # --------------------------------------------------------
    atom_enc, atom_dec, bond_enc, bond_dec = (
        get_encoders_decoders(
            cfg.chembl_smi,
            cfg.bace1_smi,
            cfg.max_atom,
        )
    )

    # --------------------------------------------------------
    # Processed dataset filenames
    # --------------------------------------------------------
    chembl_name = (
        Path(cfg.chembl_smi).stem
        + str(cfg.max_atom)
        + ".pt"
    )

    bace_name = (
        Path(cfg.bace1_smi).stem
        + str(cfg.max_atom)
        + ".pt"
    )

    # --------------------------------------------------------
    # General ChEMBL dataset
    # --------------------------------------------------------
    general = DruggenDataset(
        cfg.data_dir,
        chembl_name,
        cfg.chembl_smi,
        cfg.max_atom,
        False,
        atom_enc,
        atom_dec,
        bond_enc,
        bond_dec,
    )

    # --------------------------------------------------------
    # Attach continuous pIC50 + BBB conditions
    #
    # ConditionalDruggenDataset attaches:
    #
    #     [1.0,
    #      normalized_pIC50,
    #      normalized_logBB]
    #
    # to every BACE1 graph.
    # --------------------------------------------------------
    target = ConditionalDruggenDataset(
        "druggen_data",
        bace_name,
        cfg.bace1_smi,
        cfg.max_atom,
        False,
        atom_enc,
        atom_dec,
        bond_enc,
        bond_dec,
        labels_csv=cfg.labels_csv,
    )

    # --------------------------------------------------------
    # DataLoaders
    #
    # Standard shuffled sampling.
    #
    # IMPORTANT:
    # No WeightedRandomSampler is used.
    #
    # This experiment isolates the effect of explicit
    # pIC50 conditioning.
    # --------------------------------------------------------
    loader = DataLoader(
        general,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    target_loader = DataLoader(
        target,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # --------------------------------------------------------
    # Graph dimensions
    # --------------------------------------------------------
    m_dim = len(atom_dec)
    b_dim = len(bond_dec)

    vertexes = int(
        general[0].x.shape[0]
    )

    print()
    print("-" * 70)
    print("DATASET CONFIGURATION")
    print("-" * 70)

    print(
        f"General graphs : {len(general):,}"
    )

    print(
        f"BACE1 graphs   : {len(target):,}"
    )

    print(
        f"Maximum atoms  : {vertexes}"
    )

    print(
        f"Atom dimension : {m_dim}"
    )

    print(
        f"Bond dimension : {b_dim}"
    )

    print(
        f"Condition dim  : {CONDITION_DIM}"
    )

    print(
        "Condition      : "
        "[BACE1=1, normalized pIC50, normalized predicted logBB]"
    )

    print()
    print("pIC50 normalization:")
    print(
        f"  mean = {PIC50_MEAN:.6f}"
    )
    print(
        f"  std  = {PIC50_STD:.6f}"
    )

    print()
    print("BBB normalization:")
    print(
        f"  mean = {BBB_MEAN:.6f}"
    )
    print(
        f"  std  = {BBB_STD:.6f}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------
    G = ConditionalGenerator(
        cfg.act,
        vertexes,
        b_dim,
        m_dim,
        cfg.dropout,
        cfg.dim,
        cfg.depth,
        cfg.heads,
        cfg.mlp_ratio,
        condition_dim=CONDITION_DIM,
    ).to(device)

    D = ConditionalDiscriminator(
        cfg.act,
        vertexes,
        b_dim,
        m_dim,
        cfg.ddropout,
        cfg.dim,
        cfg.ddepth,
        cfg.heads,
        cfg.mlp_ratio,
        condition_dim=CONDITION_DIM,
    ).to(device)

    # --------------------------------------------------------
    # Optional warm start from an existing compatible experiment.
    # Optimizer states are new; only the model weights are initialized.
    # --------------------------------------------------------
    if cfg.init_generator is not None:
        load_model_checkpoint(
            G,
            cfg.init_generator,
            device,
            "generator",
        )

        if cfg.reset_critic_head:
            load_model_checkpoint(
                D,
                cfg.init_discriminator,
                device,
                "discriminator",
                exclude_prefixes=("node_mlp.",),
            )
        else:
            load_model_checkpoint(
                D,
                cfg.init_discriminator,
                device,
                "discriminator",
            )

    # --------------------------------------------------------
    # Critic-only call.
    #
    # D.forward() returns (critic_score, property_prediction).
    # The WGAN critic path does not need the property head, so it
    # is skipped here (return_property=False); the property head
    # is evaluated explicitly via D.predict_properties().
    # --------------------------------------------------------
    def critic(edge, node, condition):
        return D(edge, node, condition, return_property=False)[0]

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------
    go = torch.optim.AdamW(
        G.parameters(),
        lr=cfg.g_lr,
        betas=(
            cfg.beta1,
            cfg.beta2,
        ),
    )

    do = torch.optim.AdamW(
        D.parameters(),
        lr=cfg.d_lr,
        betas=(
            cfg.beta1,
            cfg.beta2,
        ),
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------
    out = Path(
        cfg.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Run metadata
    # --------------------------------------------------------
    info = {
        **vars(cfg),

        "device": str(device),

        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),

        "general_samples": len(general),

        "target_samples": len(target),

        "vertexes": vertexes,

        "node_dim": m_dim,

        "edge_dim": b_dim,

        "condition_dim": CONDITION_DIM,

        "condition_definition": [
            "BACE1 target = 1.0",
            "normalized pIC50",
            "normalized predicted logBB",
        ],

        "pIC50_normalization": {
            "mean": PIC50_MEAN,
            "std": PIC50_STD,
        },

        "BBB_normalization": {
            "mean": BBB_MEAN,
            "std": BBB_STD,
        },

        "affinity_sampling": {
            "enabled": False,
            "method": "standard_shuffle",
        },

        "training_pairing": {
            "method": "full_general_dataset_with_cycled_bace1",
            "general_dataset_size": len(general),
            "bace1_dataset_size": len(target),
        },

        "max_steps_per_epoch": cfg.max_steps,

        "issue3_graph_aware_property_head": {
            "enabled": True,
            "head_input": "post-Transformer node + edge (condition-free)",
            "lambda_aux": cfg.lambda_aux,
            "lambda_condition": cfg.lambda_condition,
            "property_input": cfg.property_input,
            "aux_holdout_frac": cfg.aux_holdout_frac,
            "aux_targets": [
                "normalized pIC50",
                "normalized predicted logBB",
            ],
        },
    }

    (
        out / "run_config.json"
    ).write_text(
        json.dumps(
            info,
            indent=2,
        )
    )

    print()
    print("-" * 70)
    print("RUN CONFIGURATION")
    print("-" * 70)

    print(
        json.dumps(
            info,
            indent=2,
        )
    )

    # ========================================================
    # Training
    # ========================================================
    for epoch in range(
        1,
        cfg.epochs + 1,
    ):

        G.train()
        D.train()

        generator_loss_total = 0.0
        discriminator_loss_total = 0.0

        aux_total = 0.0
        cond_total = 0.0

        aux_prop_total = torch.zeros(2, device=device)
        cond_prop_total = torch.zeros(2, device=device)

        # Per-epoch buffers for diagnostics (kept on the GPU).
        held_pred, held_true = [], []
        fake_read, fake_want = [], []

        steps = 0

        # ----------------------------------------------------
        # BACE1 target iterator
        #
        # The general ChEMBL dataset is much larger than the
        # BACE1 dataset.
        #
        # Therefore, the BACE1 loader is restarted whenever
        # it becomes exhausted.
        #
        # This ensures that every general ChEMBL batch is used.
        # ----------------------------------------------------
        target_loader_iterator = iter(
            target_loader
        )

        # ----------------------------------------------------
        # Traverse the complete general ChEMBL dataset
        # ----------------------------------------------------
        for mol_batch in loader:

            # ------------------------------------------------
            # Get the next BACE1 batch.
            #
            # Restart the BACE1 loader after it is exhausted.
            # ------------------------------------------------
            try:

                real_batch = next(
                    target_loader_iterator
                )

            except StopIteration:

                target_loader_iterator = iter(
                    target_loader
                )

                real_batch = next(
                    target_loader_iterator
                )

            # =================================================
            # General ChEMBL batch
            # =================================================
            _, mol_adj, mol_x = load_molecules(
                mol_batch,
                b_dim=b_dim,
                m_dim=m_dim,
                device=device,
                batch_size=cfg.batch_size,
            )

            # =================================================
            # Real BACE1 batch
            # =================================================
            _, real_adj, real_x = load_molecules(
                real_batch,
                b_dim=b_dim,
                m_dim=m_dim,
                device=device,
                batch_size=cfg.batch_size,
            )

            # =================================================
            # Real BACE1 condition
            #
            # Shape:
            #
            #     [batch, 3]
            #
            #     [1.0,
            #      normalized pIC50,
            #      normalized predicted logBB]
            # =================================================
            real_condition = (
                real_batch.condition
                .to(device)
                .view(
                    cfg.batch_size,
                    CONDITION_DIM,
                )
            )

            # =================================================
            # Generation condition
            #
            # The generator receives the actual continuous
            # pIC50 and BBB values represented by the BACE1
            # training batch.
            #
            # No affinity weighting is performed.
            # No binary BBB condition is used.
            # =================================================
            desired_condition = (
                real_condition.clone()
            )

            # BACE1 target condition is always 1.
            desired_condition[:, 0] = 1.0

            # =================================================
            # Discriminator update
            # =================================================
            D.zero_grad(
                set_to_none=True
            )

            # -------------------------------------------------
            # Generate fake graphs
            # -------------------------------------------------
            (
                _,
                _,
                node_sample,
                edge_sample,
            ) = G(
                mol_adj,
                mol_x,
                desired_condition,
            )

            # -------------------------------------------------
            # Real BACE1 score
            # -------------------------------------------------
            real_score = critic(
                real_adj,
                real_x,
                real_condition,
            )

            # -------------------------------------------------
            # Fake score
            #
            # The critic receives categorical graph tensors for both
            # real and fake molecules. G returns logits, so convert
            # them to a straight-through discrete graph first. In the
            # forward pass these are hard one-hot categories; in the
            # backward pass the softmax path remains differentiable.
            # -------------------------------------------------
            fake_edge_graph, fake_node_graph = relax_graph(
                node_sample,
                edge_sample,
                mode="st",
            )

            fake_score = critic(
                fake_edge_graph.detach(),
                fake_node_graph.detach(),
                desired_condition,
            )

            # =================================================
            # WGAN-GP interpolation
            # =================================================
            eps_n = torch.rand(
                cfg.batch_size,
                1,
                1,
                device=device,
            )

            eps_e = torch.rand(
                cfg.batch_size,
                1,
                1,
                1,
                device=device,
            )

            interpolated_node = (
                eps_n * real_x
                + (1.0 - eps_n) * fake_node_graph.detach()
            ).requires_grad_(True)

            interpolated_edge = (
                eps_e * real_adj
                + (1.0 - eps_e) * fake_edge_graph.detach()
            ).requires_grad_(True)

            interpolated_score = critic(
                interpolated_edge,
                interpolated_node,
                desired_condition,
            )

            # -------------------------------------------------
            # Gradient penalty
            # -------------------------------------------------
            gradients = torch.autograd.grad(
                interpolated_score,
                [
                    interpolated_node,
                    interpolated_edge,
                ],
                torch.ones_like(
                    interpolated_score
                ),
                create_graph=True,
            )

            gradient = torch.cat(
                [
                    gradients[0].flatten(1),
                    gradients[1].flatten(1),
                ],
                dim=1,
            )

            gradient_penalty = (
                (
                    gradient.norm(
                        2,
                        dim=1,
                    )
                    - 1.0
                )
                ** 2
            ).mean()

            # -------------------------------------------------
            # WGAN-GP discriminator loss
            # -------------------------------------------------
            adversarial_d_loss = (
                fake_score.mean()
                - real_score.mean()
                + cfg.lambda_gp
                * gradient_penalty
            )

            # -------------------------------------------------
            # Graph-aware auxiliary property loss (REAL graphs)
            #
            # The head reads the post-Transformer node + edge
            # representation WITHOUT any condition injected.
            # Targets are the (normalized) labels of the real
            # BACE1 molecules themselves.
            # -------------------------------------------------
            real_props = real_condition[:, 1:3]

            real_pred = D.predict_properties(
                real_adj,
                real_x,
            )

            holdout = heldout_flags(
                real_batch.smiles,
                cfg.aux_holdout_frac,
                device,
            )

            aux_per_prop = masked_mse(
                real_pred,
                real_props,
                ~holdout,
            )

            aux_loss = aux_per_prop.mean()

            discriminator_loss = (
                adversarial_d_loss
                + cfg.lambda_aux
                * aux_loss
            )

            discriminator_loss.backward()

            do.step()

            if holdout.any():
                held_pred.append(real_pred.detach()[holdout])
                held_true.append(real_props.detach()[holdout])

            # =================================================
            # Generator update
            # =================================================
            G.zero_grad(
                set_to_none=True
            )

            (
                _,
                _,
                node_sample,
                edge_sample,
            ) = G(
                mol_adj,
                mol_x,
                desired_condition,
            )

            # The generator's adversarial objective must use the same
            # categorical graph representation as the critic's real/fake
            # update. Straight-through relaxation preserves gradients to
            # the generator while presenting hard one-hot graphs forward.
            fake_edge_adv, fake_node_adv = relax_graph(
                node_sample,
                edge_sample,
                mode="st",
            )

            adversarial_g_loss = -critic(
                fake_edge_adv,
                fake_node_adv,
                desired_condition,
            ).mean()

            # -------------------------------------------------
            # Condition-consistency loss
            #
            # The generated graph is presented to the head in
            # the same discrete form that matrices2mol() will
            # decode (straight-through by default). The head's
            # reading must match the REQUESTED pIC50 / logBB.
            # -------------------------------------------------
            fake_edge_p, fake_node_p = relax_graph(
                node_sample,
                edge_sample,
                cfg.property_input,
            )

            fake_pred = D.predict_properties(
                fake_edge_p,
                fake_node_p,
            )

            wanted_props = desired_condition[:, 1:3]

            cond_per_prop = (
                (fake_pred - wanted_props) ** 2
            ).mean(dim=0)

            cond_loss = cond_per_prop.mean()

            generator_loss = (
                adversarial_g_loss
                + cfg.lambda_condition
                * cond_loss
            )

            generator_loss.backward()

            go.step()

            fake_read.append(fake_pred.detach())
            fake_want.append(wanted_props.detach())

            # -------------------------------------------------
            # Accumulate losses
            # -------------------------------------------------
            generator_loss_total += float(
                generator_loss.detach()
            )

            discriminator_loss_total += float(
                discriminator_loss.detach()
            )

            aux_total += float(aux_loss.detach())
            cond_total += float(cond_loss.detach())

            aux_prop_total += aux_per_prop.detach()
            cond_prop_total += cond_per_prop.detach()

            steps += 1

            # -------------------------------------------------
            # Optional smoke-test limit
            #
            # When --max_steps is provided, stop the epoch
            # after the requested number of steps.
            #
            # When --max_steps is omitted, the complete
            # general ChEMBL loader is traversed.
            # -------------------------------------------------
            if (
                cfg.max_steps is not None
                and steps >= cfg.max_steps
            ):
                break

        # ====================================================
        # Epoch statistics
        # ====================================================
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"G={generator_loss_total / steps:.4f} | "
            f"D={discriminator_loss_total / steps:.4f} | "
            f"Aux={aux_total / steps:.4f} | "
            f"Cond={cond_total / steps:.4f} | "
            f"Steps={steps}"
        )

        # ----------------------------------------------------
        # Diagnostics (see module docstring)
        #
        #   1. Does the head generalise to UNSEEN real molecules?
        #      (held-out R2; pIC50 is the critical column)
        #   2. Does the generator move the head's reading toward
        #      the requested value? (corr(requested, read))
        # ----------------------------------------------------
        aux_mean = (aux_prop_total / steps).tolist()
        cond_mean = (cond_prop_total / steps).tolist()

        print(
            f"    head train MSE : "
            f"pIC50={aux_mean[0]:.4f}  logBB={aux_mean[1]:.4f}"
        )

        if held_pred:

            hp = torch.cat(held_pred)
            ht = torch.cat(held_true)

            r2 = r2_per_property(hp, ht).tolist()

            print(
                f"    head HELD-OUT R2: "
                f"pIC50={r2[0]:.3f}  logBB={r2[1]:.3f}  "
                f"(n={hp.shape[0]})"
            )

        else:

            print(
                "    head HELD-OUT R2: n/a "
                "(aux_holdout_frac=0 or no held-out samples)"
            )

        fr = torch.cat(fake_read)
        fw = torch.cat(fake_want)

        corr = pearson_per_property(fr, fw).tolist()

        print(
            f"    G->head cond MSE: "
            f"pIC50={cond_mean[0]:.4f}  logBB={cond_mean[1]:.4f}"
        )

        print(
            f"    corr(requested, head reading of generated): "
            f"pIC50={corr[0]:.3f}  logBB={corr[1]:.3f}"
        )

        # ====================================================
        # Save epoch checkpoints
        # ====================================================
        torch.save(
            G.state_dict(),
            out
            / f"generator_epoch_{epoch:03d}.pt",
        )

        torch.save(
            D.state_dict(),
            out
            / f"discriminator_epoch_{epoch:03d}.pt",
        )

    # ========================================================
    # Final models
    # ========================================================
    torch.save(
        G.state_dict(),
        out / "joint_generator.pt",
    )

    torch.save(
        D.state_dict(),
        out / "joint_discriminator.pt",
    )

    # ========================================================
    # Completion
    # ========================================================
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Saved generator: "
        f"{out / 'joint_generator.pt'}"
    )

    print(
        f"Saved discriminator: "
        f"{out / 'joint_discriminator.pt'}"
    )


# ============================================================
# Command-line configuration
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--chembl_smi",
    required=True,
)


parser.add_argument(
    "--bace1_smi",
    required=True,
)


parser.add_argument(
    "--labels_csv",
    required=True,
)


parser.add_argument(
    "--data_dir",
    default="data",
)


parser.add_argument(
    "--output_dir",
    default=(
        "experiments/models/"
        "DrugGEN-BACE1-AffinityBBB-Conditioned"
    ),
)


parser.add_argument(
    "--init_generator",
    default=None,
    help=(
        "Optional compatible generator state_dict to warm-start from. "
        "Use together with --init_discriminator."
    ),
)


parser.add_argument(
    "--init_discriminator",
    default=None,
    help=(
        "Optional compatible discriminator state_dict to warm-start from. "
        "Use together with --init_generator."
    ),
)


parser.add_argument(
    "--reset_critic_head",
    action="store_true",
    help=(
        "When warm-starting, retain discriminator shared layers and its "
        "auxiliary property head, but freshly initialize node_mlp, the "
        "Wasserstein critic scoring head. Requires both initialization checkpoints."
    ),
)


parser.add_argument(
    "--max_atom",
    type=int,
    default=45,
)


parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
)


parser.add_argument(
    "--epochs",
    type=int,
    default=10,
)


parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help=(
        "Maximum training steps per epoch. "
        "Use for smoke tests; default=None means full epoch."
    ),
)


parser.add_argument(
    "--dim",
    type=int,
    default=128,
)


parser.add_argument(
    "--depth",
    type=int,
    default=1,
)


parser.add_argument(
    "--ddepth",
    type=int,
    default=1,
)


parser.add_argument(
    "--heads",
    type=int,
    default=8,
)


parser.add_argument(
    "--mlp_ratio",
    type=int,
    default=3,
)


parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
)


parser.add_argument(
    "--ddropout",
    type=float,
    default=0.0,
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
    "--beta1",
    type=float,
    default=0.9,
)


parser.add_argument(
    "--beta2",
    type=float,
    default=0.999,
)


parser.add_argument(
    "--lambda_gp",
    type=float,
    default=10.0,
)


parser.add_argument(
    "--act",
    default="relu",
    choices=[
        "relu",
        "leaky",
        "sigmoid",
        "tanh",
    ],
)


parser.add_argument(
    "--seed",
    type=int,
    default=1,
)


parser.add_argument(
    "--lambda_aux",
    type=float,
    default=1.0,
    help="Weight of the auxiliary property loss on REAL graphs (D step).",
)


parser.add_argument(
    "--lambda_condition",
    type=float,
    default=1.0,
    help="Weight of the generator condition-consistency loss.",
)


parser.add_argument(
    "--property_input",
    default="st",
    choices=[
        "st",
        "softmax",
        "logits",
    ],
    help=(
        "How generated graphs are shown to the property head. "
        "st = straight-through discrete graph (matches decoding, default); "
        "softmax = relaxed probabilities; logits = raw logits (not advised)."
    ),
)


parser.add_argument(
    "--aux_holdout_frac",
    type=float,
    default=0.1,
    help=(
        "Fraction of BACE1 graphs excluded from the AUXILIARY loss only "
        "(deterministic by SMILES) to measure head generalisation. "
        "0 disables the hold-out."
    ),
)


if __name__ == "__main__":

    args = parser.parse_args()

    train(args)