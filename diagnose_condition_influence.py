
"""
Issue 3: CONDITION-INFLUENCE DIAGNOSTIC

Purpose
-------
Determine whether the continuous BACE1 pIC50 condition actually influences
the ConditionalGenerator internally.

We compare two conditions:

    Condition A: pIC50 = 5.5
    Condition B: pIC50 = 8.0

The following are kept IDENTICAL:

    - input ChEMBL graphs
    - random seed
    - latent noise

Only the requested pIC50 condition changes.

We measure:

    1. condition embedding difference
    2. node representation difference before Transformer
    3. edge representation difference before Transformer
    4. Transformer node output difference
    5. Transformer edge output difference
    6. node-logit difference
    7. edge-logit difference

Interpretation
--------------
If condition embedding changes but later representations do not:
    condition influence is being lost inside the generator.

If node/edge representations change but logits barely change:
    the Transformer/readout is suppressing the condition effect.

If logits change substantially:
    conditioning works internally and the problem is probably later,
    such as graph decoding or the learned mapping itself.

No retraining is performed.
No model weights are changed.
"""


# =====================================================================
# IMPORTS
# =====================================================================

from pathlib import Path

import random

import numpy as np

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

CHEMBL_SMI = (
    "data/chembl_train.smi"
)

BACE1_SMI = (
    "druggen_data/bace1.smi"
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

SEED = 42


# =====================================================================
# NORMALIZATION
# =====================================================================

PIC50_MEAN = 6.691315

PIC50_STD = 1.308678

BBB_MEAN = -0.077873

BBB_STD = 0.751623


# =====================================================================
# CONDITIONS
# =====================================================================

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

        torch.cuda.manual_seed_all(
            seed
        )


# =====================================================================
# CHECKPOINT
# =====================================================================

def extract_state_dict(
    checkpoint
):

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            return checkpoint[
                "model_state_dict"
            ]

        if (
            "state_dict"
            in checkpoint
        ):

            return checkpoint[
                "state_dict"
            ]

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
# CONDITION VECTOR
# =====================================================================

def make_condition(
    pic50,
):

    normalized_pic50 = (

        pic50
        - PIC50_MEAN
    ) / PIC50_STD

    normalized_logbb = (

        LOGBB
        - BBB_MEAN
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
# STATISTICS
# =====================================================================

def tensor_stats(
    name,
    tensor,
):

    tensor = tensor.detach()

    print(
        f"\n{name}"
    )

    print(
        f"  shape : "
        f"{tuple(tensor.shape)}"
    )

    print(
        f"  mean  : "
        f"{tensor.mean().item():.8f}"
    )

    print(
        f"  std   : "
        f"{tensor.std().item():.8f}"
    )

    print(
        f"  min   : "
        f"{tensor.min().item():.8f}"
    )

    print(
        f"  max   : "
        f"{tensor.max().item():.8f}"
    )

    print(
        f"  abs mean : "
        f"{tensor.abs().mean().item():.8f}"
    )


# =====================================================================
# DIFFERENCE STATISTICS
# =====================================================================

def difference_stats(
    name,
    a,
    b,
):

    difference = (
        a - b
    ).detach()

    mean_abs = (
        difference.abs()
        .mean()
        .item()
    )

    max_abs = (
        difference.abs()
        .max()
        .item()
    )

    rms = torch.sqrt(
        torch.mean(
            difference ** 2
        )
    ).item()

    a_norm = (
        torch.norm(a)
        .item()
    )

    relative = (
        torch.norm(difference)
        / (
            torch.norm(a)
            + 1e-12
        )
    ).item()

    print(
        f"\n{name}"
    )

    print(
        f"  mean absolute difference : "
        f"{mean_abs:.10f}"
    )

    print(
        f"  max absolute difference  : "
        f"{max_abs:.10f}"
    )

    print(
        f"  RMS difference           : "
        f"{rms:.10f}"
    )

    print(
        f"  reference norm            : "
        f"{a_norm:.10f}"
    )

    print(
        f"  relative difference       : "
        f"{relative:.10f}"
    )

    return {

        "mean_abs":
            mean_abs,

        "max_abs":
            max_abs,

        "rms":
            rms,

        "relative":
            relative,
    }


# =====================================================================
# GENERATOR INTERNAL FORWARD
# =====================================================================

@torch.no_grad()
def inspect_generator(
    generator,
    a_tensor,
    x_tensor,
    condition,
):

    # ---------------------------------------------------------------
    # Condition embedding
    # ---------------------------------------------------------------

    condition_embedding = (
        generator.condition_layers(
            condition
        )
        .to(
            x_tensor.dtype
        )
    )

    # ---------------------------------------------------------------
    # Original DrugGEN node/edge transformations
    # ---------------------------------------------------------------

    node = (
        generator.node_layers(
            x_tensor
        )
    )

    edge = (
        generator.edge_layers(
            a_tensor
        )
    )

    # ---------------------------------------------------------------
    # Condition injection
    # ---------------------------------------------------------------

    node_conditioned = (

        node
        + condition_embedding
        .unsqueeze(1)
    )

    edge_conditioned = (

        edge
        + condition_embedding
        .unsqueeze(1)
        .unsqueeze(1)
    )

    # ---------------------------------------------------------------
    # Edge symmetrization
    # ---------------------------------------------------------------

    edge_conditioned = (

        edge_conditioned

        + edge_conditioned.permute(
            0,
            2,
            1,
            3,
        )

    ) / 2

    # ---------------------------------------------------------------
    # Transformer
    # ---------------------------------------------------------------

    transformer_node, transformer_edge = (

        generator.TransformerEncoder(

            node_conditioned,

            edge_conditioned,
        )
    )

    # ---------------------------------------------------------------
    # Readout
    # ---------------------------------------------------------------

    node_logits = (
        generator.readout_n(
            transformer_node
        )
    )

    edge_logits = (
        generator.readout_e(
            transformer_edge
        )
    )

    return {

        "condition_embedding":
            condition_embedding,

        "node_before_condition":
            node,

        "edge_before_condition":
            edge,

        "node_after_condition":
            node_conditioned,

        "edge_after_condition":
            edge_conditioned,

        "transformer_node":
            transformer_node,

        "transformer_edge":
            transformer_edge,

        "node_logits":
            node_logits,

        "edge_logits":
            edge_logits,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():

    # =================================================================
    # SEED
    # =================================================================

    set_seed(
        SEED
    )

    print(
        "=" * 80
    )

    print(
        "ISSUE 3: CONDITION-INFLUENCE DIAGNOSTIC"
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
        "Generator checkpoint:"
    )

    print(
        f"  {GENERATOR_CHECKPOINT}"
    )

    print()

    print(
        "Conditions:"
    )

    print(
        f"  A: pIC50 = {PIC50_A}"
    )

    print(
        f"  B: pIC50 = {PIC50_B}"
    )

    print(
        f"  Both logBB = {LOGBB}"
    )

    print()

    # =================================================================
    # ENCODERS / DECODERS
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

    print(
        "Dataset:"
    )

    print(
        f"  graphs : "
        f"{len(dataset)}"
    )

    print(
        f"  nodes  : "
        f"{nodes}"
    )

    print(
        f"  edges  : "
        f"{edges}"
    )

    print()

    # =================================================================
    # LOAD GENERATOR
    # =================================================================

    generator = load_generator(

        nodes,

        edges,
    )

    print(
        "Generator loaded successfully."
    )

    print()

    # =================================================================
    # FIXED INPUT GRAPH
    # =================================================================

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        drop_last=True,
    )

    batch = next(
        iter(loader)
    )

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

    print(
        "Fixed input tensors:"
    )

    print(
        f"  node input : "
        f"{tuple(x_tensor.shape)}"
    )

    print(
        f"  edge input : "
        f"{tuple(a_tensor.shape)}"
    )

    print()

    # =================================================================
    # CREATE CONDITIONS
    # =================================================================

    condition_a_single = make_condition(
        PIC50_A
    )

    condition_b_single = make_condition(
        PIC50_B
    )

    condition_a = (

        condition_a_single

        .unsqueeze(0)

        .repeat(
            BATCH_SIZE,
            1,
        )
    )

    condition_b = (

        condition_b_single

        .unsqueeze(0)

        .repeat(
            BATCH_SIZE,
            1,
        )
    )

    print(
        "Condition A:"
    )

    print(
        condition_a_single
        .detach()
        .cpu()
        .numpy()
    )

    print()

    print(
        "Condition B:"
    )

    print(
        condition_b_single
        .detach()
        .cpu()
        .numpy()
    )

    print()

    # =================================================================
    # VERIFY CONDITION DIFFERENCE
    # =================================================================

    difference_stats(

        "INPUT CONDITION DIFFERENCE",

        condition_a,

        condition_b,
    )

    # =================================================================
    # USE IDENTICAL RANDOM STATE
    # =================================================================

    """
    We need the same stochastic latent noise for both conditions.

    Save RNG states before the first generator call, restore them before
    the second generator call.

    This isolates the effect of the condition itself.
    """

    cpu_rng_state = (
        torch.get_rng_state()
    )

    if torch.cuda.is_available():

        cuda_rng_state = (
            torch.cuda.get_rng_state(
                DEVICE
            )
        )

    # =================================================================
    # CONDITION A
    # =================================================================

    print()
    print(
        "=" * 80
    )

    print(
        f"RUN A: requested pIC50 = {PIC50_A}"
    )

    print(
        "=" * 80
    )

    output_a = inspect_generator(

        generator,

        a_tensor,

        x_tensor,

        condition_a,
    )

    # =================================================================
    # RESTORE RNG
    # =================================================================

    torch.set_rng_state(
        cpu_rng_state
    )

    if torch.cuda.is_available():

        torch.cuda.set_rng_state(
            cuda_rng_state,
            DEVICE,
        )

    # =================================================================
    # CONDITION B
    # =================================================================

    print()
    print(
        "=" * 80
    )

    print(
        f"RUN B: requested pIC50 = {PIC50_B}"
    )

    print(
        "=" * 80
    )

    output_b = inspect_generator(

        generator,

        a_tensor,

        x_tensor,

        condition_b,
    )

    # =================================================================
    # PRINT INDIVIDUAL TENSOR STATISTICS
    # =================================================================

    print()
    print(
        "=" * 80
    )

    print(
        "CONDITION A INTERNAL STATISTICS"
    )

    print(
        "=" * 80
    )

    tensor_stats(

        "Condition embedding A",

        output_a[
            "condition_embedding"
        ],
    )

    tensor_stats(

        "Node before condition A",

        output_a[
            "node_before_condition"
        ],
    )

    tensor_stats(

        "Edge before condition A",

        output_a[
            "edge_before_condition"
        ],
    )

    tensor_stats(

        "Node after condition A",

        output_a[
            "node_after_condition"
        ],
    )

    tensor_stats(

        "Edge after condition A",

        output_a[
            "edge_after_condition"
        ],
    )

    tensor_stats(

        "Transformer node A",

        output_a[
            "transformer_node"
        ],
    )

    tensor_stats(

        "Transformer edge A",

        output_a[
            "transformer_edge"
        ],
    )

    tensor_stats(

        "Node logits A",

        output_a[
            "node_logits"
        ],
    )

    tensor_stats(

        "Edge logits A",

        output_a[
            "edge_logits"
        ],
    )

    # =================================================================
    # CONDITION B STATISTICS
    # =================================================================

    print()
    print(
        "=" * 80
    )

    print(
        "CONDITION B INTERNAL STATISTICS"
    )

    print(
        "=" * 80
    )

    tensor_stats(

        "Condition embedding B",

        output_b[
            "condition_embedding"
        ],
    )

    tensor_stats(

        "Node after condition B",

        output_b[
            "node_after_condition"
        ],
    )

    tensor_stats(

        "Edge after condition B",

        output_b[
            "edge_after_condition"
        ],
    )

    tensor_stats(

        "Transformer node B",

        output_b[
            "transformer_node"
        ],
    )

    tensor_stats(

        "Transformer edge B",

        output_b[
            "transformer_edge"
        ],
    )

    tensor_stats(

        "Node logits B",

        output_b[
            "node_logits"
        ],
    )

    tensor_stats(

        "Edge logits B",

        output_b[
            "edge_logits"
        ],
    )

    # =================================================================
    # CONDITION DIFFERENCES
    # =================================================================

    print()
    print(
        "=" * 80
    )

    print(
        "CONDITION-TO-OUTPUT DIFFERENCES"
    )

    print(
        "=" * 80
    )

    results = {}

    results[
        "condition_embedding"
    ] = difference_stats(

        "1. Condition embedding",

        output_a[
            "condition_embedding"
        ],

        output_b[
            "condition_embedding"
        ],
    )

    results[
        "node_after_condition"
    ] = difference_stats(

        "2. Node representation after condition",

        output_a[
            "node_after_condition"
        ],

        output_b[
            "node_after_condition"
        ],
    )

    results[
        "edge_after_condition"
    ] = difference_stats(

        "3. Edge representation after condition",

        output_a[
            "edge_after_condition"
        ],

        output_b[
            "edge_after_condition"
        ],
    )

    results[
        "transformer_node"
    ] = difference_stats(

        "4. Transformer node output",

        output_a[
            "transformer_node"
        ],

        output_b[
            "transformer_node"
        ],
    )

    results[
        "transformer_edge"
    ] = difference_stats(

        "5. Transformer edge output",

        output_a[
            "transformer_edge"
        ],

        output_b[
            "transformer_edge"
        ],
    )

    results[
        "node_logits"
    ] = difference_stats(

        "6. Node logits",

        output_a[
            "node_logits"
        ],

        output_b[
            "node_logits"
        ],
    )

    results[
        "edge_logits"
    ] = difference_stats(

        "7. Edge logits",

        output_a[
            "edge_logits"
        ],

        output_b[
            "edge_logits"
        ],
    )

    # =================================================================
    # FINAL INTERPRETATION
    # =================================================================

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL DIAGNOSTIC INTERPRETATION"
    )

    print(
        "=" * 80
    )

    condition_change = (
        results[
            "condition_embedding"
        ]["relative"]
    )

    node_change = (
        results[
            "node_after_condition"
        ]["relative"]
    )

    edge_change = (
        results[
            "edge_after_condition"
        ]["relative"]
    )

    transformer_node_change = (
        results[
            "transformer_node"
        ]["relative"]
    )

    transformer_edge_change = (
        results[
            "transformer_edge"
        ]["relative"]
    )

    node_logit_change = (
        results[
            "node_logits"
        ]["relative"]
    )

    edge_logit_change = (
        results[
            "edge_logits"
        ]["relative"]
    )

    print()

    print(
        "Relative changes:"
    )

    print(
        f"  Condition embedding : "
        f"{condition_change:.8f}"
    )

    print(
        f"  Node after condition: "
        f"{node_change:.8f}"
    )

    print(
        f"  Edge after condition: "
        f"{edge_change:.8f}"
    )

    print(
        f"  Transformer node    : "
        f"{transformer_node_change:.8f}"
    )

    print(
        f"  Transformer edge    : "
        f"{transformer_edge_change:.8f}"
    )

    print(
        f"  Node logits         : "
        f"{node_logit_change:.8f}"
    )

    print(
        f"  Edge logits         : "
        f"{edge_logit_change:.8f}"
    )

    print()

    # =================================================================
    # AUTOMATIC DIAGNOSIS
    # =================================================================

    if (
        condition_change > 1e-3
        and node_logit_change < 1e-4
        and edge_logit_change < 1e-4
    ):

        print(
            "DIAGNOSIS:"
        )

        print(
            "The condition embedding changes, but "
            "the generator output logits are almost "
            "unchanged."
        )

        print(
            "The conditioning signal is therefore "
            "being suppressed before the final graph "
            "readout."
        )

    elif (
        condition_change > 1e-3
        and (
            node_logit_change >= 1e-4
            or edge_logit_change >= 1e-4
        )
    ):

        print(
            "DIAGNOSIS:"
        )

        print(
            "The pIC50 condition DOES influence the "
            "generator logits."
        )

        print(
            "The next investigation should therefore "
            "focus on graph decoding and the learned "
            "relationship between generated chemistry "
            "and BACE1 activity."
        )

    else:

        print(
            "DIAGNOSIS:"
        )

        print(
            "The condition embedding itself is not "
            "changing as expected."
        )

        print(
            "The condition construction/pathway should "
            "be inspected before retraining."
        )

    print()

    print(
        "No retraining was performed."
    )

    print(
        "=" * 80
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()
