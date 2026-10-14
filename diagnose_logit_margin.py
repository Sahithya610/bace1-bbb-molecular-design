"""
ISSUE 3 — LOGIT MARGIN DIAGNOSTIC

Purpose
-------
Determine whether the continuous pIC50 condition is strong enough
to cross categorical node/bond decision boundaries.

We compare:

    Condition A: pIC50 = 5.5
    Condition B: pIC50 = 8.0

using the SAME input graphs.

No retraining is performed.
No model weights are modified.

For every node and bond position we calculate:

    top-1 logit
    top-2 logit
    top-1 - top-2 margin

We also measure the condition-induced change in logits.

Interpretation:

    Large margin + small condition effect
        -> condition cannot change the categorical class.

    Small margin + comparable condition effect
        -> condition can potentially change chemistry.

    Large number of class flips
        -> conditioning is reaching the categorical output.
"""


# =====================================================================
# IMPORTS
# =====================================================================

from pathlib import Path

import random

import numpy as np

import pandas as pd

import torch

from torch_geometric.loader import DataLoader

from src.data.dataset import DruggenDataset

from src.data.utils import (
    get_encoders_decoders,
    load_molecules,
)

from src.model.conditional_models import (
    ConditionalGenerator,
)


# =====================================================================
# DEVICE
# =====================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# PATHS
# =====================================================================

GENERATOR_CHECKPOINT = Path(
    "experiments/models/Issue3-Diagnostic/generator_epoch_002.pt"
)

CHEMBL_SMI = "data/chembl_train.smi"

BACE1_SMI = "druggen_data/bace1.smi"

OUTPUT_CSV = Path(
    "results/issue3_logit_margin.csv"
)


# =====================================================================
# SETTINGS
# =====================================================================

VERTEXES = 45

DIM = 128

DEPTH = 1

HEADS = 8

MLP_RATIO = 3

DROPOUT = 0.0

BATCH_SIZE = 32

NUM_BATCHES = 4

SEED = 42


# =====================================================================
# NORMALIZATION
# =====================================================================

PIC50_MEAN = 6.691315

PIC50_STD = 1.308678

BBB_MEAN = -0.077873

BBB_STD = 0.751623


PIC50_A = 5.5

PIC50_B = 8.0

LOGBB = 0.0


# =====================================================================
# SEED
# =====================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =====================================================================
# CHECKPOINT
# =====================================================================

def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:

            return checkpoint["state_dict"]

    return checkpoint


# =====================================================================
# LOAD GENERATOR
# =====================================================================

def load_generator(
    nodes,
    edges,
):

    generator = ConditionalGenerator(

        act="relu",

        vertexes=VERTEXES,

        edges=edges,

        nodes=nodes,

        dropout=DROPOUT,

        dim=DIM,

        depth=DEPTH,

        heads=HEADS,

        mlp_ratio=MLP_RATIO,
    ).to(DEVICE)

    checkpoint = torch.load(

        GENERATOR_CHECKPOINT,

        map_location=DEVICE,

        weights_only=False,
    )

    generator.load_state_dict(

        extract_state_dict(
            checkpoint
        )
    )

    generator.eval()

    return generator


# =====================================================================
# CONDITION
# =====================================================================

def make_condition(
    pic50,
):

    normalized_pic50 = (

        pic50 - PIC50_MEAN

    ) / PIC50_STD

    normalized_logbb = (

        LOGBB - BBB_MEAN

    ) / BBB_STD

    return torch.tensor(

        [
            1.0,
            normalized_pic50,
            normalized_logbb,
        ],

        dtype=torch.float32,

        device=DEVICE,
    )


# =====================================================================
# GENERATOR FORWARD
# =====================================================================

@torch.no_grad()
def run_generator(
    generator,
    a_tensor,
    x_tensor,
    condition,
):

    output = generator(

        a_tensor,

        x_tensor,

        condition,
    )

    if not isinstance(
        output,
        (tuple, list)
    ):

        raise RuntimeError(
            "Unexpected generator output type: "
            f"{type(output)}"
        )

    if len(output) < 4:

        raise RuntimeError(
            "Unexpected generator output length: "
            f"{len(output)}"
        )

    node_latent = output[0]

    edge_latent = output[1]

    node_logits = output[2]

    edge_logits = output[3]

    return (

        node_latent,

        edge_latent,

        node_logits,

        edge_logits,
    )


# =====================================================================
# TOP-1 / TOP-2 MARGIN
# =====================================================================

def calculate_margin(logits):

    """
    logits shape:

        Node:
            [B, 45, 14]

        Edge:
            [B, 45, 45, 5]

    Returns:

        top1
        top2
        margin
        predicted_class
    """

    values, indices = torch.topk(

        logits,

        k=2,

        dim=-1,
    )

    top1 = values[..., 0]

    top2 = values[..., 1]

    margin = top1 - top2

    predicted_class = indices[..., 0]

    return (

        top1,

        top2,

        margin,

        predicted_class,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    set_seed(SEED)

    print(
        "=" * 80
    )

    print(
        "ISSUE 3 — LOGIT MARGIN DIAGNOSTIC"
    )

    print(
        "=" * 80
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()

    print(
        "Checkpoint:"
    )

    print(
        f"  {GENERATOR_CHECKPOINT}"
    )

    print()

    print(
        f"Condition A pIC50: {PIC50_A}"
    )

    print(
        f"Condition B pIC50: {PIC50_B}"
    )

    print(
        f"Both logBB: {LOGBB}"
    )

    # =================================================================
    # ENCODERS
    # =================================================================

    (
        atom_encoder,
        atom_decoder,
        bond_encoder,
        bond_decoder,
    ) = get_encoders_decoders(

        CHEMBL_SMI,

        BACE1_SMI,

        VERTEXES,
    )

    # =================================================================
    # DATASET
    # =================================================================

    dataset = DruggenDataset(

        root="data",

        dataset_file="chembl_train45.pt",

        raw_files=[
            "chembl_train.smi",
            "pre_filter.pt",
            "pre_transform.pt",
        ],

        max_atom=VERTEXES,

        features=False,

        atom_encoder=atom_encoder,

        atom_decoder=atom_decoder,

        bond_encoder=bond_encoder,

        bond_decoder=bond_decoder,
    )

    nodes = int(
        dataset[0].x.shape[1]
    )

    edges = len(
        bond_decoder
    )

    print()

    print(
        "Dataset:"
    )

    print(
        f"  Graphs: {len(dataset)}"
    )

    print(
        f"  Node classes: {nodes}"
    )

    print(
        f"  Edge classes: {edges}"
    )

    # =================================================================
    # GENERATOR
    # =================================================================

    generator = load_generator(

        nodes,

        edges,
    )

    print()

    print(
        "Generator loaded successfully."
    )

    # =================================================================
    # CONDITIONS
    # =================================================================

    condition_a = make_condition(
        PIC50_A
    )

    condition_b = make_condition(
        PIC50_B
    )

    print()

    print(
        "Condition A:"
    )

    print(
        condition_a
        .detach()
        .cpu()
        .numpy()
    )

    print()

    print(
        "Condition B:"
    )

    print(
        condition_b
        .detach()
        .cpu()
        .numpy()
    )

    # =================================================================
    # LOADER
    # =================================================================

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        drop_last=True,
    )

    # =================================================================
    # STORAGE
    # =================================================================

    node_margins_a = []

    node_margins_b = []

    edge_margins_a = []

    edge_margins_b = []

    node_logit_changes = []

    edge_logit_changes = []

    node_class_flips = 0

    edge_class_flips = 0

    node_positions = 0

    edge_positions = 0

    node_near_boundary_a = 0

    node_near_boundary_b = 0

    edge_near_boundary_a = 0

    edge_near_boundary_b = 0

    # Thresholds for "near decision boundary"

    NODE_MARGIN_THRESHOLD = 0.10

    EDGE_MARGIN_THRESHOLD = 0.10

    # =================================================================
    # PROCESS
    # =================================================================

    for batch_number, batch in enumerate(loader):

        if batch_number >= NUM_BATCHES:

            break

        print()

        print(
            "-" * 80
        )

        print(
            f"Batch "
            f"{batch_number + 1}/"
            f"{NUM_BATCHES}"
        )

        print(
            "-" * 80
        )

        # -------------------------------------------------------------
        # GRAPH TENSORS
        # -------------------------------------------------------------

        (
            _,
            a_tensor,
            x_tensor,
        ) = load_molecules(

            data=batch,

            batch_size=BATCH_SIZE,

            device=DEVICE,

            b_dim=edges,

            m_dim=nodes,
        )

        # -------------------------------------------------------------
        # SAME RNG
        # -------------------------------------------------------------

        cpu_rng_state = (
            torch.get_rng_state()
        )

        if torch.cuda.is_available():

            cuda_rng_state = (
                torch.cuda.get_rng_state(
                    DEVICE
                )
            )

        # -------------------------------------------------------------
        # CONDITION A
        # -------------------------------------------------------------

        condition_a_batch = (

            condition_a

            .unsqueeze(0)

            .repeat(
                BATCH_SIZE,
                1,
            )
        )

        (
            _,
            _,
            node_logits_a,
            edge_logits_a,
        ) = run_generator(

            generator,

            a_tensor,

            x_tensor,

            condition_a_batch,
        )

        # -------------------------------------------------------------
        # RESTORE RNG
        # -------------------------------------------------------------

        torch.set_rng_state(
            cpu_rng_state
        )

        if torch.cuda.is_available():

            torch.cuda.set_rng_state(
                cuda_rng_state,
                DEVICE,
            )

        # -------------------------------------------------------------
        # CONDITION B
        # -------------------------------------------------------------

        condition_b_batch = (

            condition_b

            .unsqueeze(0)

            .repeat(
                BATCH_SIZE,
                1,
            )
        )

        (
            _,
            _,
            node_logits_b,
            edge_logits_b,
        ) = run_generator(

            generator,

            a_tensor,

            x_tensor,

            condition_b_batch,
        )

        # -------------------------------------------------------------
        # NODE MARGINS
        # -------------------------------------------------------------

        (
            node_top1_a,
            node_top2_a,
            node_margin_a,
            node_class_a,
        ) = calculate_margin(
            node_logits_a
        )

        (
            node_top1_b,
            node_top2_b,
            node_margin_b,
            node_class_b,
        ) = calculate_margin(
            node_logits_b
        )

        # -------------------------------------------------------------
        # EDGE MARGINS
        # -------------------------------------------------------------

        (
            edge_top1_a,
            edge_top2_a,
            edge_margin_a,
            edge_class_a,
        ) = calculate_margin(
            edge_logits_a
        )

        (
            edge_top1_b,
            edge_top2_b,
            edge_margin_b,
            edge_class_b,
        ) = calculate_margin(
            edge_logits_b
        )

        # -------------------------------------------------------------
        # LOGIT DIFFERENCES
        # -------------------------------------------------------------

        node_diff = torch.abs(

            node_logits_b

            - node_logits_a
        )

        edge_diff = torch.abs(

            edge_logits_b

            - edge_logits_a
        )

        # -------------------------------------------------------------
        # NODE STATISTICS
        # -------------------------------------------------------------

        node_margins_a.extend(

            node_margin_a
            .detach()
            .cpu()
            .flatten()
            .numpy()
            .tolist()
        )

        node_margins_b.extend(

            node_margin_b
            .detach()
            .cpu()
            .flatten()
            .numpy()
            .tolist()
        )

        node_logit_changes.extend(

            node_diff
            .detach()
            .cpu()
            .flatten()
            .numpy()
            .tolist()
        )

        node_class_flips += int(

            (
                node_class_a
                != node_class_b
            )
            .sum()
            .item()
        )

        node_positions += (

            node_class_a
            .numel()
        )

        node_near_boundary_a += int(

            (
                node_margin_a
                < NODE_MARGIN_THRESHOLD
            )
            .sum()
            .item()
        )

        node_near_boundary_b += int(

            (
                node_margin_b
                < NODE_MARGIN_THRESHOLD
            )
            .sum()
            .item()
        )

        # -------------------------------------------------------------
        # EDGE STATISTICS
        # -------------------------------------------------------------

        edge_margins_a.extend(

            edge_margin_a
            .detach()
            .cpu()
            .flatten()
            .numpy()
            .tolist()
        )

        edge_margins_b.extend(

            edge_margin_b
            .detach()
            .cpu()
            .flatten()
            .numpy()
            .tolist()
        )

        edge_logit_changes.extend(

            edge_diff
            .detach()
            .cpu()
            .flatten()
            .numpy()
            .tolist()
        )

        edge_class_flips += int(

            (
                edge_class_a
                != edge_class_b
            )
            .sum()
            .item()
        )

        edge_positions += (

            edge_class_a
            .numel()
        )

        edge_near_boundary_a += int(

            (
                edge_margin_a
                < EDGE_MARGIN_THRESHOLD
            )
            .sum()
            .item()
        )

        edge_near_boundary_b += int(

            (
                edge_margin_b
                < EDGE_MARGIN_THRESHOLD
            )
            .sum()
            .item()
        )

        # -------------------------------------------------------------
        # BATCH SUMMARY
        # -------------------------------------------------------------

        print()

        print(
            "Node:"
        )

        print(
            "  mean margin A: "
            f"{node_margin_a.mean().item():.6f}"
        )

        print(
            "  mean margin B: "
            f"{node_margin_b.mean().item():.6f}"
        )

        print(
            "  mean |logit change|: "
            f"{node_diff.mean().item():.6f}"
        )

        print(
            "  class flips: "
            f"{(
                node_class_a != node_class_b
            ).sum().item()}"
        )

        print()

        print(
            "Edge:"
        )

        print(
            "  mean margin A: "
            f"{edge_margin_a.mean().item():.6f}"
        )

        print(
            "  mean margin B: "
            f"{edge_margin_b.mean().item():.6f}"
        )

        print(
            "  mean |logit change|: "
            f"{edge_diff.mean().item():.6f}"
        )

        print(
            "  class flips: "
            f"{(
                edge_class_a != edge_class_b
            ).sum().item()}"
        )

    # =================================================================
    # CONVERT
    # =================================================================

    node_margins_a = np.asarray(
        node_margins_a
    )

    node_margins_b = np.asarray(
        node_margins_b
    )

    edge_margins_a = np.asarray(
        edge_margins_a
    )

    edge_margins_b = np.asarray(
        edge_margins_b
    )

    node_logit_changes = np.asarray(
        node_logit_changes
    )

    edge_logit_changes = np.asarray(
        edge_logit_changes
    )

    # =================================================================
    # SUMMARY
    # =================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL LOGIT MARGIN SUMMARY"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "NODE OUTPUT"
    )

    print(
        f"  Positions tested: "
        f"{node_positions}"
    )

    print(
        f"  Mean margin A: "
        f"{node_margins_a.mean():.6f}"
    )

    print(
        f"  Median margin A: "
        f"{np.median(node_margins_a):.6f}"
    )

    print(
        f"  Mean margin B: "
        f"{node_margins_b.mean():.6f}"
    )

    print(
        f"  Median margin B: "
        f"{np.median(node_margins_b):.6f}"
    )

    print(
        f"  Mean |logit change|: "
        f"{node_logit_changes.mean():.6f}"
    )

    print(
        f"  Median |logit change|: "
        f"{np.median(node_logit_changes):.6f}"
    )

    print(
        f"  Class flips: "
        f"{node_class_flips}"
    )

    print(
        f"  Flip percentage: "
        f"{100.0 * node_class_flips / node_positions:.6f}%"
    )

    print(
        f"  Near-boundary A (<0.10): "
        f"{node_near_boundary_a}"
    )

    print(
        f"  Near-boundary B (<0.10): "
        f"{node_near_boundary_b}"
    )

    print(
        f"  Near-boundary percentage A: "
        f"{100.0 * node_near_boundary_a / node_positions:.6f}%"
    )

    print(
        f"  Near-boundary percentage B: "
        f"{100.0 * node_near_boundary_b / node_positions:.6f}%"
    )

    print()

    print(
        "EDGE OUTPUT"
    )

    print(
        f"  Positions tested: "
        f"{edge_positions}"
    )

    print(
        f"  Mean margin A: "
        f"{edge_margins_a.mean():.6f}"
    )

    print(
        f"  Median margin A: "
        f"{np.median(edge_margins_a):.6f}"
    )

    print(
        f"  Mean margin B: "
        f"{edge_margins_b.mean():.6f}"
    )

    print(
        f"  Median margin B: "
        f"{np.median(edge_margins_b):.6f}"
    )

    print(
        f"  Mean |logit change|: "
        f"{edge_logit_changes.mean():.6f}"
    )

    print(
        f"  Median |logit change|: "
        f"{np.median(edge_logit_changes):.6f}"
    )

    print(
        f"  Class flips: "
        f"{edge_class_flips}"
    )

    print(
        f"  Flip percentage: "
        f"{100.0 * edge_class_flips / edge_positions:.6f}%"
    )

    print(
        f"  Near-boundary A (<0.10): "
        f"{edge_near_boundary_a}"
    )

    print(
        f"  Near-boundary B (<0.10): "
        f"{edge_near_boundary_b}"
    )

    print(
        f"  Near-boundary percentage A: "
        f"{100.0 * edge_near_boundary_a / edge_positions:.6f}%"
    )

    print(
        f"  Near-boundary percentage B: "
        f"{100.0 * edge_near_boundary_b / edge_positions:.6f}%"
    )

    # =================================================================
    # CONDITION EFFECT RATIO
    # =================================================================

    node_margin_mean = (
        node_margins_a.mean()
        +
        node_margins_b.mean()
    ) / 2.0

    edge_margin_mean = (
        edge_margins_a.mean()
        +
        edge_margins_b.mean()
    ) / 2.0

    node_effect_ratio = (

        node_logit_changes.mean()

        / max(
            node_margin_mean,
            1e-12,
        )
    )

    edge_effect_ratio = (

        edge_logit_changes.mean()

        / max(
            edge_margin_mean,
            1e-12,
        )
    )

    print()

    print(
        "CONDITION EFFECT VS DECISION MARGIN"
    )

    print()

    print(
        f"Node mean condition effect / "
        f"mean margin: "
        f"{node_effect_ratio:.6f}"
    )

    print(
        f"Edge mean condition effect / "
        f"mean margin: "
        f"{edge_effect_ratio:.6f}"
    )

    # =================================================================
    # SAVE SUMMARY
    # =================================================================

    summary = pd.DataFrame({

        "metric": [

            "node_positions",

            "node_mean_margin_A",

            "node_median_margin_A",

            "node_mean_margin_B",

            "node_median_margin_B",

            "node_mean_abs_logit_change",

            "node_median_abs_logit_change",

            "node_class_flips",

            "node_flip_percentage",

            "node_near_boundary_A",

            "node_near_boundary_B",

            "node_near_boundary_percentage_A",

            "node_near_boundary_percentage_B",

            "edge_positions",

            "edge_mean_margin_A",

            "edge_median_margin_A",

            "edge_mean_margin_B",

            "edge_median_margin_B",

            "edge_mean_abs_logit_change",

            "edge_median_abs_logit_change",

            "edge_class_flips",

            "edge_flip_percentage",

            "edge_near_boundary_A",

            "edge_near_boundary_B",

            "edge_near_boundary_percentage_A",

            "edge_near_boundary_percentage_B",

            "node_effect_to_margin_ratio",

            "edge_effect_to_margin_ratio",
        ],

        "value": [

            node_positions,

            node_margins_a.mean(),

            np.median(node_margins_a),

            node_margins_b.mean(),

            np.median(node_margins_b),

            node_logit_changes.mean(),

            np.median(node_logit_changes),

            node_class_flips,

            100.0
            * node_class_flips
            / node_positions,

            node_near_boundary_a,

            node_near_boundary_b,

            100.0
            * node_near_boundary_a
            / node_positions,

            100.0
            * node_near_boundary_b
            / node_positions,

            edge_positions,

            edge_margins_a.mean(),

            np.median(edge_margins_a),

            edge_margins_b.mean(),

            np.median(edge_margins_b),

            edge_logit_changes.mean(),

            np.median(edge_logit_changes),

            edge_class_flips,

            100.0
            * edge_class_flips
            / edge_positions,

            edge_near_boundary_a,

            edge_near_boundary_b,

            100.0
            * edge_near_boundary_a
            / edge_positions,

            100.0
            * edge_near_boundary_b
            / edge_positions,

            node_effect_ratio,

            edge_effect_ratio,
        ],
    })

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()

    print(
        "Summary saved:"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    # =================================================================
    # INTERPRETATION
    # =================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "DIAGNOSTIC INTERPRETATION"
    )

    print(
        "=" * 80
    )

    if (
        node_class_flips == 0
        and edge_class_flips == 0
    ):

        print()

        print(
            "DIAGNOSIS:"
        )

        print(
            "The pIC50 condition changes the continuous "
            "logits but does not cross categorical "
            "decision boundaries."
        )

        print()

        print(
            "The next architectural decision depends "
            "on the ratio between condition-induced "
            "logit changes and categorical margins."
        )

    elif (
        node_class_flips > 0
        or edge_class_flips > 0
    ):

        print()

        print(
            "DIAGNOSIS:"
        )

        print(
            "The pIC50 condition crosses categorical "
            "decision boundaries for at least some "
            "graph components."
        )

        print()

        print(
            "Therefore the conditioning mechanism is "
            "capable of changing discrete graph chemistry."
        )

    print()

    print(
        "No retraining was performed."
    )

    print()

    print(
        "=" * 80
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()
