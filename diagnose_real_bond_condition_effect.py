
"""
ISSUE 3 — REAL-BOND vs NO-BOND CONDITION EFFECT DIAGNOSTIC

Purpose
-------
Determine whether the pIC50 condition affects chemically meaningful
bond positions or mainly no-bond / padding positions.

Comparison:

    Condition A: pIC50 = 5.5
    Condition B: pIC50 = 8.0

The SAME input graphs and SAME random generator state are used.

For each edge position we report:

1. Whether the input graph contains a real bond.
2. Top-1 / top-2 categorical margin.
3. Condition-induced logit change.
4. Whether the predicted categorical class changes.
5. Whether the position is near a categorical decision boundary.

No retraining is performed.
No model weights are modified.
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

from src.model.conditional_models import ConditionalGenerator


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
    "results/issue3_real_bond_condition_effect.csv"
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

MARGIN_THRESHOLD = 0.10


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

def load_generator(nodes, edges):

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

def make_condition(pic50):

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
            "Unexpected generator output type."
        )

    if len(output) < 4:

        raise RuntimeError(
            "Unexpected generator output length."
        )

    return (

        output[0],  # node latent

        output[1],  # edge latent

        output[2],  # node logits

        output[3],  # edge logits
    )


# =====================================================================
# TOP-1 / TOP-2
# =====================================================================

def categorical_statistics(logits):

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
# DETERMINE REAL BONDS
# =====================================================================

def get_real_bond_mask(
    a_tensor,
    batch_size,
    vertices,
):

    """
    DrugGEN edge tensor:

        [B, V, V, edge_features]

    The final edge feature corresponds to the no-bond
    category in the DrugGEN representation.

    A position is treated as a real-bond position when
    its decoded input edge class is NOT the no-bond class.

    We therefore use argmax over the edge feature dimension.
    """

    input_edge_class = torch.argmax(

        a_tensor,

        dim=-1,
    )

    no_bond_class = (

        a_tensor.shape[-1] - 1
    )

    real_bond_mask = (

        input_edge_class
        != no_bond_class
    )

    # Ignore diagonal entries because self-bonds
    # are not chemically meaningful.

    diagonal = torch.eye(

        vertices,

        dtype=torch.bool,

        device=a_tensor.device,
    ).unsqueeze(0)

    diagonal = diagonal.expand(

        batch_size,

        -1,

        -1,
    )

    real_bond_mask = (

        real_bond_mask
        & ~diagonal
    )

    return (

        real_bond_mask,

        input_edge_class,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    set_seed(SEED)

    print("=" * 80)

    print(
        "ISSUE 3 — REAL-BOND vs NO-BOND "
        "CONDITION EFFECT DIAGNOSTIC"
    )

    print("=" * 80)

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
        "Generator checkpoint:"
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

    print("Dataset:")

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

    print("Condition A:")

    print(
        condition_a
        .detach()
        .cpu()
        .numpy()
    )

    print()

    print("Condition B:")

    print(
        condition_b
        .detach()
        .cpu()
        .numpy()
    )

    # =================================================================
    # DATA LOADER
    # =================================================================

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        drop_last=True,
    )

    # =================================================================
    # AGGREGATE STORAGE
    # =================================================================

    stats = {

        "real": {

            "positions": 0,

            "near_a": 0,

            "near_b": 0,

            "class_flips": 0,

            "logit_change_sum": 0.0,

            "margin_a_sum": 0.0,

            "margin_b_sum": 0.0,
        },

        "no_bond": {

            "positions": 0,

            "near_a": 0,

            "near_b": 0,

            "class_flips": 0,

            "logit_change_sum": 0.0,

            "margin_a_sum": 0.0,

            "margin_b_sum": 0.0,
        },
    }

    # =================================================================
    # PROCESS BATCHES
    # =================================================================

    for batch_number, batch in enumerate(loader):

        if batch_number >= NUM_BATCHES:

            break

        print()

        print("-" * 80)

        print(
            f"Batch "
            f"{batch_number + 1}/"
            f"{NUM_BATCHES}"
        )

        print("-" * 80)

        # -------------------------------------------------------------
        # LOAD GRAPH TENSORS
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
        # REAL-BOND MASK
        # -------------------------------------------------------------

        real_bond_mask, input_edge_class = (
            get_real_bond_mask(

                a_tensor,

                BATCH_SIZE,

                VERTEXES,
            )
        )

        no_bond_mask = ~real_bond_mask

        # -------------------------------------------------------------
        # CONDITIONS
        # -------------------------------------------------------------

        condition_a_batch = (

            condition_a

            .unsqueeze(0)

            .repeat(
                BATCH_SIZE,
                1,
            )
        )

        condition_b_batch = (

            condition_b

            .unsqueeze(0)

            .repeat(
                BATCH_SIZE,
                1,
            )
        )

        # -------------------------------------------------------------
        # SAME RANDOM INPUT
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

        (
            _,
            _,
            _,
            edge_logits_a,
        ) = run_generator(

            generator,

            a_tensor,

            x_tensor,

            condition_a_batch,
        )

        # -------------------------------------------------------------
        # RESTORE RANDOM STATE
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

        (
            _,
            _,
            _,
            edge_logits_b,
        ) = run_generator(

            generator,

            a_tensor,

            x_tensor,

            condition_b_batch,
        )

        # -------------------------------------------------------------
        # CATEGORICAL STATISTICS
        # -------------------------------------------------------------

        (
            _,
            _,
            margin_a,
            class_a,
        ) = categorical_statistics(

            edge_logits_a
        )

        (
            _,
            _,
            margin_b,
            class_b,
        ) = categorical_statistics(

            edge_logits_b
        )

        # -------------------------------------------------------------
        # CONDITION EFFECT
        # -------------------------------------------------------------

        logit_change = torch.abs(

            edge_logits_b

            - edge_logits_a
        )

        # -------------------------------------------------------------
        # REAL-BOND POSITIONS
        # -------------------------------------------------------------

        for name, mask in [

            ("real", real_bond_mask),

            ("no_bond", no_bond_mask),
        ]:

            masked_margins_a = (
                margin_a[mask]
            )

            masked_margins_b = (
                margin_b[mask]
            )

            masked_change = (
                logit_change[mask]
            )

            masked_class_a = (
                class_a[mask]
            )

            masked_class_b = (
                class_b[mask]
            )

            count = (
                masked_margins_a.numel()
            )

            if count == 0:

                continue

            class_flips = int(

                (
                    masked_class_a
                    != masked_class_b
                )
                .sum()
                .item()
            )

            near_a = int(

                (
                    masked_margins_a
                    < MARGIN_THRESHOLD
                )
                .sum()
                .item()
            )

            near_b = int(

                (
                    masked_margins_b
                    < MARGIN_THRESHOLD
                )
                .sum()
                .item()
            )

            stats[name][
                "positions"
            ] += count

            stats[name][
                "near_a"
            ] += near_a

            stats[name][
                "near_b"
            ] += near_b

            stats[name][
                "class_flips"
            ] += class_flips

            stats[name][
                "logit_change_sum"
            ] += (
                masked_change.sum()
                .item()
            )

            stats[name][
                "margin_a_sum"
            ] += (
                masked_margins_a
                .sum()
                .item()
            )

            stats[name][
                "margin_b_sum"
            ] += (
                masked_margins_b
                .sum()
                .item()
            )

            print()

            print(
                name.upper()
                + ":"
            )

            print(
                "  positions: "
                f"{count}"
            )

            print(
                "  mean margin A: "
                f"{masked_margins_a.mean().item():.6f}"
            )

            print(
                "  mean margin B: "
                f"{masked_margins_b.mean().item():.6f}"
            )

            print(
                "  mean |logit change|: "
                f"{masked_change.mean().item():.6f}"
            )

            print(
                "  near-boundary A: "
                f"{near_a}"
                f" ({100.0 * near_a / count:.4f}%)"
            )

            print(
                "  near-boundary B: "
                f"{near_b}"
                f" ({100.0 * near_b / count:.4f}%)"
            )

            print(
                "  class flips: "
                f"{class_flips}"
                f" ({100.0 * class_flips / count:.4f}%)"
            )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print()

    print("=" * 80)

    print(
        "FINAL REAL-BOND vs NO-BOND SUMMARY"
    )

    print("=" * 80)

    rows = []

    for name in [
        "real",
        "no_bond",
    ]:

        s = stats[name]

        count = s["positions"]

        if count == 0:

            continue

        mean_margin_a = (

            s["margin_a_sum"]

            / count
        )

        mean_margin_b = (

            s["margin_b_sum"]

            / count
        )

        mean_change = (

            s["logit_change_sum"]

            / count
        )

        average_margin = (

            mean_margin_a
            +
            mean_margin_b

        ) / 2.0

        effect_ratio = (

            mean_change

            / max(
                average_margin,
                1e-12,
            )
        )

        near_a_pct = (

            100.0
            * s["near_a"]
            / count
        )

        near_b_pct = (

            100.0
            * s["near_b"]
            / count
        )

        flip_pct = (

            100.0
            * s["class_flips"]
            / count
        )

        print()

        print(
            name.upper()
        )

        print(
            f"  Positions: {count}"
        )

        print(
            f"  Mean margin A: "
            f"{mean_margin_a:.6f}"
        )

        print(
            f"  Mean margin B: "
            f"{mean_margin_b:.6f}"
        )

        print(
            f"  Mean |logit change|: "
            f"{mean_change:.6f}"
        )

        print(
            f"  Effect / margin: "
            f"{effect_ratio:.6f}"
        )

        print(
            f"  Near-boundary A: "
            f"{s['near_a']} "
            f"({near_a_pct:.6f}%)"
        )

        print(
            f"  Near-boundary B: "
            f"{s['near_b']} "
            f"({near_b_pct:.6f}%)"
        )

        print(
            f"  Class flips: "
            f"{s['class_flips']} "
            f"({flip_pct:.6f}%)"
        )

        rows.append({

            "position_type": name,

            "positions": count,

            "mean_margin_A": mean_margin_a,

            "mean_margin_B": mean_margin_b,

            "mean_abs_logit_change": mean_change,

            "effect_to_margin_ratio": effect_ratio,

            "near_boundary_A": s["near_a"],

            "near_boundary_A_pct": near_a_pct,

            "near_boundary_B": s["near_b"],

            "near_boundary_B_pct": near_b_pct,

            "class_flips": s["class_flips"],

            "class_flip_pct": flip_pct,
        })

    # =================================================================
    # SAVE
    # =================================================================

    OUTPUT_CSV.parent.mkdir(

        parents=True,

        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(

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

    print("=" * 80)

    print(
        "INTERPRETATION"
    )

    print("=" * 80)

    real_stats = stats["real"]

    no_bond_stats = stats["no_bond"]

    if real_stats["positions"] > 0:

        real_change = (

            real_stats["logit_change_sum"]

            / real_stats["positions"]
        )

        real_margin = (

            real_stats["margin_a_sum"]

            +
            real_stats["margin_b_sum"]

        ) / (

            2.0
            * real_stats["positions"]
        )

        real_ratio = (

            real_change

            / max(
                real_margin,
                1e-12,
            )
        )

        print()

        print(
            f"Real-bond condition effect / "
            f"margin: {real_ratio:.6f}"
        )

    if no_bond_stats["positions"] > 0:

        no_bond_change = (

            no_bond_stats["logit_change_sum"]

            / no_bond_stats["positions"]
        )

        no_bond_margin = (

            no_bond_stats["margin_a_sum"]

            +
            no_bond_stats["margin_b_sum"]

        ) / (

            2.0
            * no_bond_stats["positions"]
        )

        no_bond_ratio = (

            no_bond_change

            / max(
                no_bond_margin,
                1e-12,
            )
        )

        print()

        print(
            f"No-bond condition effect / "
            f"margin: {no_bond_ratio:.6f}"
        )

    print()

    if real_stats["class_flips"] > 0:

        print(
            "DIAGNOSIS:"
        )

        print(
            "The pIC50 condition changes at least "
            "some chemically meaningful bond "
            "categories."
        )

        print(
            "The conditioning pathway therefore "
            "has demonstrated chemistry-level influence."
        )

    elif (

        real_stats["near_b"] > 0

        and real_stats["positions"] > 0

    ):

        print(
            "DIAGNOSIS:"
        )

        print(
            "The pIC50 condition affects real-bond "
            "positions but does not cross their "
            "categorical boundaries."
        )

        print(
            "This supports investigating a stronger "
            "chemistry-level conditioning mechanism."
        )

    else:

        print(
            "DIAGNOSIS:"
        )

        print(
            "The pIC50 condition does not materially "
            "change categorical decisions at real-bond "
            "positions in this checkpoint."
        )

        print(
            "The condition effect is therefore likely "
            "being expressed primarily away from "
            "chemically meaningful connectivity."
        )

    print()

    print(
        "No retraining was performed."
    )

    print()

    print("=" * 80)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()

