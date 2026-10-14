"""
Issue 3 diagnostic: property head vs independent BACE1 predictor.

No retraining.
No architecture change.

Purpose
-------
Test whether the continuous BACE1 pIC50 condition actually changes the
generated chemistry.

Requested conditions:

    pIC50 = [5.5, 6.5, 7.0, 8.0, 9.0]
    logBB = 0.0

The exact same fixed ChEMBL input graphs are used for every condition.

For every condition, compare:

1. Issue 3 auxiliary property head
2. Independent BACE1 ExtraTrees predictor

Important tensor distinction
----------------------------
ConditionalGenerator returns:

    z_n          -> [B, 45, 128]
    z_e          -> [B, 45, 45, 128]
    node_logits  -> [B, 45, 14]
    edge_logits  -> [B, 45, 45, 5]

The discriminator property path expects ORIGINAL graph features:

    node features = 14
    edge features = 5

Therefore z_n/z_e MUST NOT be passed to predict_properties().

Instead:

    node_logits
    edge_logits
          |
          v
    relax_graph(..., mode="st")
          |
          v
    relaxed node/edge graph
          |
          v
    discriminator.predict_properties()

Molecule decoding uses:

    dataset.matrices2mol(..., strict=False)

No model weights are changed.
No training is performed.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules

from src.model.conditional_models import (
    ConditionalGenerator,
    ConditionalDiscriminator,
    relax_graph,
)


# =====================================================================
# DEVICE
# =====================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================================
# PATHS
# =====================================================================

GENERATOR_CHECKPOINT = Path(
    "experiments/models/Issue3-Diagnostic/generator_epoch_002.pt"
)

DISCRIMINATOR_CHECKPOINT = Path(
    "experiments/models/Issue3-Diagnostic/discriminator_epoch_002.pt"
)

BACE1_MODEL = Path(
    "results/bace1_activity_model/bace1_activity_model.joblib"
)

CHEMBL_SMI = "data/chembl_train.smi"

BACE1_SMI = "druggen_data/bace1.smi"

OUTPUT = Path(
    "results/issue3_head_vs_external_diagnostic.csv"
)


# =====================================================================
# CONDITIONS
# =====================================================================

REQUESTED_PIC50 = [
    5.5,
    6.5,
    7.0,
    8.0,
    9.0,
]

REQUESTED_LOGBB = 0.0


# =====================================================================
# DIAGNOSTIC SETTINGS
# =====================================================================

BATCH_SIZE = 256

NUM_FIXED_BATCHES = 4

SEED = 42


# =====================================================================
# DRUGGEN ARCHITECTURE
# =====================================================================

VERTEXES = 45

DIM = 128

DEPTH = 1

HEADS = 8

MLP_RATIO = 3

DROPOUT = 0.0

DDROPOUT = 0.0


# =====================================================================
# NORMALIZATION
# =====================================================================

PIC50_MEAN = 6.691315

PIC50_STD = 1.308678

BBB_MEAN = -0.077873

BBB_STD = 0.751623


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# CHECKPOINT UTILITIES
# =====================================================================

def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    return checkpoint


# =====================================================================
# MODEL LOADING
# =====================================================================

def load_generator(vertexes, nodes, edges):

    model = ConditionalGenerator(
        act="relu",
        vertexes=vertexes,
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

    model.load_state_dict(
        extract_state_dict(checkpoint)
    )

    model.eval()

    return model


def load_discriminator(vertexes, nodes, edges):

    model = ConditionalDiscriminator(
        act="relu",
        vertexes=vertexes,
        edges=edges,
        nodes=nodes,
        dropout=DDROPOUT,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_ratio=MLP_RATIO,
    ).to(DEVICE)

    checkpoint = torch.load(
        DISCRIMINATOR_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        extract_state_dict(checkpoint)
    )

    model.eval()

    return model


def load_external_bace1_model():

    import joblib

    return joblib.load(
        BACE1_MODEL
    )


# =====================================================================
# MOLECULE DECODING
# =====================================================================

def canonical_smiles(mol):

    if mol is None:
        return None

    try:

        Chem.SanitizeMol(mol)

        return Chem.MolToSmiles(
            mol,
            canonical=True,
        )

    except Exception:

        return None


def clean_smiles(smiles):

    if smiles is None:
        return None

    # Keep largest connected component.
    if "." in smiles:

        fragments = smiles.split(".")

        smiles = max(
            fragments,
            key=len,
        )

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            return None

        smiles = canonical_smiles(
            mol
        )

    # Handle wildcard atoms.
    if smiles is not None and "*" in smiles:

        smiles = smiles.replace(
            "*",
            "C",
        )

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            return None

        smiles = canonical_smiles(
            mol
        )

    return smiles


def decode_graphs(
    dataset,
    generated_nodes,
    generated_edges,
):

    smiles_list = []

    decoder_failures = 0

    for edge_graph, node_graph in zip(
        generated_edges,
        generated_nodes,
    ):

        try:

            mol = dataset.matrices2mol(
                node_graph.detach()
                .cpu()
                .numpy(),

                edge_graph.detach()
                .cpu()
                .numpy(),

                strict=False,
            )

            smiles = canonical_smiles(
                mol
            )

            smiles = clean_smiles(
                smiles
            )

            smiles_list.append(
                smiles
            )

        except Exception as exc:

            decoder_failures += 1

            print(
                "Decoder warning: "
                f"{type(exc).__name__}: {exc}"
            )

            smiles_list.append(
                None
            )

    return (
        smiles_list,
        decoder_failures,
    )


# =====================================================================
# MORGAN FINGERPRINTS
# =====================================================================

def smiles_to_morgan(smiles_list):

    fingerprints = []

    valid_smiles = []

    for smiles in smiles_list:

        if smiles is None:
            continue

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=2048,
        )

        fingerprints.append(
            np.asarray(
                fp,
                dtype=np.float32,
            )
        )

        valid_smiles.append(
            smiles
        )

    return (
        valid_smiles,
        fingerprints,
    )


# =====================================================================
# ONE CONDITION
# =====================================================================

@torch.no_grad()
def run_condition(
    generator,
    discriminator,
    external_model,
    dataset,
    fixed_batches,
    requested_pic50,
    requested_logbb,
    nodes,
    edges,
):

    # ---------------------------------------------------------------
    # Normalize requested condition.
    # ---------------------------------------------------------------

    normalized_condition = torch.tensor(
        [
            1.0,

            (
                requested_pic50
                - PIC50_MEAN
            ) / PIC50_STD,

            (
                requested_logbb
                - BBB_MEAN
            ) / BBB_STD,
        ],

        dtype=torch.float32,

        device=DEVICE,
    )

    all_smiles = []

    all_head_predictions = []

    total_generated = 0

    total_decoder_failures = 0

    # ---------------------------------------------------------------
    # Generate using the same fixed input graphs.
    # ---------------------------------------------------------------

    for mol_batch in fixed_batches:

        total_generated += (
            mol_batch.num_graphs
        )

        # -----------------------------------------------------------
        # Convert PyG batch into DrugGEN tensors.
        # -----------------------------------------------------------

        _, a_tensor, x_tensor = load_molecules(
            data=mol_batch,

            batch_size=mol_batch.num_graphs,

            device=DEVICE,

            b_dim=edges,

            m_dim=nodes,
        )

        # -----------------------------------------------------------
        # Repeat requested condition for entire batch.
        # -----------------------------------------------------------

        condition = (
            normalized_condition
            .unsqueeze(0)
            .repeat(
                mol_batch.num_graphs,
                1,
            )
        )

        # -----------------------------------------------------------
        # Generator forward.
        # -----------------------------------------------------------

        (
            z_e,
            z_n,
            node_logits,
            edge_logits,
        ) = generator(
            a_tensor,
            x_tensor,
            condition,
        )

        # -----------------------------------------------------------
        # z_e and z_n are intentionally NOT used for the
        # property-head prediction below.
        #
        # They are 128-dimensional latent representations.
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        # Decode generated graph.
        # -----------------------------------------------------------

        generated_nodes = torch.argmax(
            node_logits,
            dim=-1,
        )

        generated_edges = torch.argmax(
            edge_logits,
            dim=-1,
        )

        (
            batch_smiles,
            decoder_failures,
        ) = decode_graphs(
            dataset,

            generated_nodes,

            generated_edges,
        )

        total_decoder_failures += (
            decoder_failures
        )

        # -----------------------------------------------------------
        # IMPORTANT:
        #
        # Convert generator logits into the graph representation
        # used by Issue 3's auxiliary condition loss.
        #
        # This gives:
        #
        # relaxed_nodes -> [B,45,14]
        # relaxed_edges -> [B,45,45,5]
        #
        # These are the correct input dimensions for the
        # discriminator's node_layers and edge_layers.
        # -----------------------------------------------------------

        relaxed_edges, relaxed_nodes = relax_graph(
            node_logits,

            edge_logits,

            mode="st",
        )

        # -----------------------------------------------------------
        # Explicit dimensional checks.
        # -----------------------------------------------------------

        if relaxed_nodes.shape[-1] != nodes:

            raise RuntimeError(
                "Unexpected node feature dimension.\n"
                f"Received: {relaxed_nodes.shape}\n"
                f"Expected final dimension: {nodes}"
            )

        if relaxed_edges.shape[-1] != edges:

            raise RuntimeError(
                "Unexpected edge feature dimension.\n"
                f"Received: {relaxed_edges.shape}\n"
                f"Expected final dimension: {edges}"
            )

        # -----------------------------------------------------------
        # Issue 3 property head.
        #
        # CORRECT:
        #
        #     relaxed_edges
        #     relaxed_nodes
        #
        # NOT:
        #
        #     z_e
        #     z_n
        # -----------------------------------------------------------

        property_prediction = (
            discriminator.predict_properties(
                relaxed_edges,
                relaxed_nodes,
            )
            .detach()
            .cpu()
            .numpy()
        )

        # -----------------------------------------------------------
        # Keep only molecules that successfully decoded.
        # -----------------------------------------------------------

        for smiles, prediction in zip(
            batch_smiles,
            property_prediction,
        ):

            if smiles is None:
                continue

            all_smiles.append(
                smiles
            )

            # Convert normalized property prediction back to
            # physical pIC50 and logBB.
            all_head_predictions.append(
                [
                    (
                        prediction[0]
                        * PIC50_STD
                        + PIC50_MEAN
                    ),

                    (
                        prediction[1]
                        * BBB_STD
                        + BBB_MEAN
                    ),
                ]
            )

    # =================================================================
    # INDEPENDENT BACE1 PREDICTION
    # =================================================================

    (
        external_valid_smiles,
        fingerprints,
    ) = smiles_to_morgan(
        all_smiles
    )

    if fingerprints:

        external_predictions = (
            external_model.predict(
                np.asarray(
                    fingerprints,
                    dtype=np.float32,
                )
            )
        )

    else:

        external_predictions = np.asarray(
            [],
            dtype=np.float64,
        )

    # =================================================================
    # PROPERTY HEAD RESULTS
    # =================================================================

    head = np.asarray(
        all_head_predictions,
        dtype=np.float64,
    )

    # =================================================================
    # RETURN RESULTS
    # =================================================================

    return {

        "requested_pIC50":
            requested_pic50,

        "requested_logBB":
            requested_logbb,

        "total_generated":
            total_generated,

        "valid":
            len(all_smiles),

        "unique":
            len(set(all_smiles)),

        "decoder_failures":
            total_decoder_failures,

        "property_head_predictions":
            len(head),

        "external_predictions":
            len(external_predictions),

        # -------------------------------------------------------------
        # Issue 3 property head
        # -------------------------------------------------------------

        "head_pIC50_mean":
            (
                float(
                    head[:, 0].mean()
                )
                if len(head)
                else np.nan
            ),

        "head_pIC50_median":
            (
                float(
                    np.median(
                        head[:, 0]
                    )
                )
                if len(head)
                else np.nan
            ),

        "head_pIC50_max":
            (
                float(
                    head[:, 0].max()
                )
                if len(head)
                else np.nan
            ),

        "head_logBB_mean":
            (
                float(
                    head[:, 1].mean()
                )
                if len(head)
                else np.nan
            ),

        "head_logBB_median":
            (
                float(
                    np.median(
                        head[:, 1]
                    )
                )
                if len(head)
                else np.nan
            ),

        "head_logBB_max":
            (
                float(
                    head[:, 1].max()
                )
                if len(head)
                else np.nan
            ),

        # -------------------------------------------------------------
        # Independent BACE1 ExtraTrees
        # -------------------------------------------------------------

        "external_pIC50_mean":
            (
                float(
                    np.mean(
                        external_predictions
                    )
                )
                if len(external_predictions)
                else np.nan
            ),

        "external_pIC50_median":
            (
                float(
                    np.median(
                        external_predictions
                    )
                )
                if len(external_predictions)
                else np.nan
            ),

        "external_pIC50_max":
            (
                float(
                    np.max(
                        external_predictions
                    )
                )
                if len(external_predictions)
                else np.nan
            ),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():

    set_seed(
        SEED
    )

    print("=" * 80)

    print(
        "ISSUE 3: PROPERTY HEAD vs "
        "INDEPENDENT BACE1 DIAGNOSTIC"
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

    print(
        "Discriminator checkpoint:"
    )

    print(
        f"  {DISCRIMINATOR_CHECKPOINT}"
    )

    print(
        "Independent BACE1 model:"
    )

    print(
        f"  {BACE1_MODEL}"
    )

    print()

    print(
        f"Requested pIC50 values: "
        f"{REQUESTED_PIC50}"
    )

    print(
        f"Requested logBB: "
        f"{REQUESTED_LOGBB}"
    )

    print()

    # =================================================================
    # LOAD ENCODERS / DECODERS
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
    # LOAD GENERAL CHEMBL DATASET
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
        "DrugGEN graph dimensions:"
    )

    print(
        f"  ChEMBL graphs : "
        f"{len(dataset)}"
    )

    print(
        f"  vertices      : "
        f"{VERTEXES}"
    )

    print(
        f"  node features : "
        f"{nodes}"
    )

    print(
        f"  edge features : "
        f"{edges}"
    )

    print()

    # =================================================================
    # DIMENSION SANITY CHECK
    # =================================================================

    if nodes != 14:

        raise RuntimeError(
            "Unexpected DrugGEN node dimension. "
            f"Expected 14, received {nodes}."
        )

    if edges != 5:

        raise RuntimeError(
            "Unexpected DrugGEN edge dimension. "
            f"Expected 5, received {edges}."
        )

    # =================================================================
    # LOAD MODELS
    # =================================================================

    print(
        "Loading Issue 3 generator..."
    )

    generator = load_generator(
        VERTEXES,
        nodes,
        edges,
    )

    print(
        "Loading Issue 3 discriminator "
        "and property head..."
    )

    discriminator = load_discriminator(
        VERTEXES,
        nodes,
        edges,
    )

    print(
        "Loading independent BACE1 "
        "ExtraTrees predictor..."
    )

    external_model = (
        load_external_bace1_model()
    )

    print()

    # =================================================================
    # FIXED INPUT DATA
    # =================================================================

    loader = DataLoader(
        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        drop_last=True,
    )

    fixed_batches = []

    for batch in loader:

        fixed_batches.append(
            batch
        )

        if (
            len(fixed_batches)
            >= NUM_FIXED_BATCHES
        ):
            break

    fixed_graph_count = (
        len(fixed_batches)
        * BATCH_SIZE
    )

    print(
        "Fixed ChEMBL input:"
    )

    print(
        f"  batches : "
        f"{len(fixed_batches)}"
    )

    print(
        f"  graphs  : "
        f"{fixed_graph_count}"
    )

    print()

    print(
        "The exact same fixed input "
        "graphs will be used for every "
        "requested pIC50 condition."
    )

    print()

    # =================================================================
    # RUN CONDITIONS
    # =================================================================

    rows = []

    for requested_pic50 in REQUESTED_PIC50:

        print("-" * 80)

        print(
            f"Requested pIC50 = "
            f"{requested_pic50:.1f}"
        )

        print(
            f"Requested logBB = "
            f"{REQUESTED_LOGBB:.1f}"
        )

        result = run_condition(

            generator=generator,

            discriminator=discriminator,

            external_model=external_model,

            dataset=dataset,

            fixed_batches=fixed_batches,

            requested_pic50=requested_pic50,

            requested_logbb=REQUESTED_LOGBB,

            nodes=nodes,

            edges=edges,
        )

        rows.append(
            result
        )

        print()

        print(
            f"Generated: "
            f"{result['total_generated']}"
        )

        print(
            f"Valid: "
            f"{result['valid']}"
        )

        print(
            f"Unique: "
            f"{result['unique']}"
        )

        print(
            f"Decoder failures: "
            f"{result['decoder_failures']}"
        )

        print()

        print(
            "Issue 3 property head:"
        )

        print(
            f"  pIC50 mean   = "
            f"{result['head_pIC50_mean']:.4f}"
        )

        print(
            f"  pIC50 median = "
            f"{result['head_pIC50_median']:.4f}"
        )

        print(
            f"  pIC50 max    = "
            f"{result['head_pIC50_max']:.4f}"
        )

        print(
            f"  logBB mean   = "
            f"{result['head_logBB_mean']:.4f}"
        )

        print()

        print(
            "Independent BACE1 predictor:"
        )

        print(
            f"  pIC50 mean   = "
            f"{result['external_pIC50_mean']:.4f}"
        )

        print(
            f"  pIC50 median = "
            f"{result['external_pIC50_median']:.4f}"
        )

        print(
            f"  pIC50 max    = "
            f"{result['external_pIC50_max']:.4f}"
        )

    # =================================================================
    # SAVE RESULTS
    # =================================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print()

    print("=" * 80)

    print(
        "FINAL DIAGNOSTIC SUMMARY"
    )

    print("=" * 80)

    print(
        df.to_string(
            index=False
        )
    )

    print()

    print(
        "Saved:"
    )

    print(
        f"  {OUTPUT}"
    )

    print()

    print("=" * 80)

    print(
        "DIAGNOSTIC INTERPRETATION"
    )

    print("=" * 80)

    print(
        "The important comparison is the change in "
        "predicted pIC50 as the requested pIC50 "
        "increases from 5.5 to 9.0."
    )

    print()

    print(
        "If the Issue 3 property-head pIC50 changes "
        "strongly while the independent BACE1 predictor "
        "remains flat, the auxiliary head may be responding "
        "to the conditioning pathway without the chemistry "
        "actually becoming more BACE1-active."
    )

    print()

    print(
        "If BOTH the property head and the independent "
        "BACE1 predictor remain approximately flat, the "
        "continuous BACE1 conditioning pathway is not "
        "producing a meaningful response."
    )

    print()

    print(
        "No retraining was performed."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()