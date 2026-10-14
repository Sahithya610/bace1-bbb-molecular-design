
"""
ISSUE 3 — CHEMICAL CONDITION EFFECT DIAGNOSTIC

Purpose
-------
Determine whether the continuous BACE1 pIC50 condition produces
genuine chemical changes after graph decoding.

Comparison:

    Condition A: pIC50 = 5.5
    Condition B: pIC50 = 8.0

The following are kept identical:

    - generator checkpoint
    - input ChEMBL graphs
    - random seed / stochastic state

Only the requested pIC50 condition changes.

Measures:

    - atom-class changes
    - chemically meaningful atom changes
    - bond-class changes
    - chemically meaningful bond changes
    - atom-count changes
    - molecular formula changes
    - heavy-atom-count changes
    - canonical SMILES changes
    - validity

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

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

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
    "results/issue3_chemical_condition_effect.csv"
)


# =====================================================================
# MODEL SETTINGS
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
# CONDITION NORMALIZATION
# =====================================================================

PIC50_MEAN = 6.691315

PIC50_STD = 1.308678

BBB_MEAN = -0.077873

BBB_STD = 0.751623


PIC50_A = 5.5

PIC50_B = 8.0

LOGBB = 0.0


# =====================================================================
# RANDOM SEED
# =====================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =====================================================================
# CHECKPOINT STATE
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
    """
    IMPORTANT

    Do NOT manually reproduce the generator's internal
    tensor operations here.

    The current ConditionalGenerator implementation
    expects its inputs in the exact format used by its
    own forward() method.

    Therefore this diagnostic calls the trained model
    directly.
    """

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

    # Current ConditionalGenerator returns:
    #
    #   node_latent
    #   edge_latent
    #   node_logits
    #   edge_logits

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
# ARGMAX GRAPH
# =====================================================================

def argmax_graph(
    node_logits,
    edge_logits,
):

    node_classes = torch.argmax(
        node_logits,
        dim=-1,
    )

    edge_classes = torch.argmax(
        edge_logits,
        dim=-1,
    )

    return (
        node_classes,
        edge_classes,
    )


# =====================================================================
# RDKit GRAPH DECODER
# =====================================================================

def decode_smiles(
    node_classes,
    edge_classes,
    atom_decoder,
    bond_decoder,
):

    try:

        node_classes = (

            node_classes

            .detach()

            .cpu()

            .numpy()
        )

        edge_classes = (

            edge_classes

            .detach()

            .cpu()

            .numpy()
        )

        mol = Chem.RWMol()

        atom_indices = {}

        # -------------------------------------------------------------
        # ADD ATOMS
        # -------------------------------------------------------------

        for i, atom_class in enumerate(
            node_classes
        ):

            atom_class = int(
                atom_class
            )

            if atom_class >= len(
                atom_decoder
            ):

                continue

            atom_symbol = atom_decoder[
                atom_class
            ]

            if atom_symbol in (
                None,
                "",
                "None",
            ):

                continue

            try:

                atom = Chem.Atom(
                    atom_symbol
                )

                atom_index = mol.AddAtom(
                    atom
                )

                atom_indices[i] = (
                    atom_index
                )

            except Exception:

                continue

        # -------------------------------------------------------------
        # ADD BONDS
        # -------------------------------------------------------------

        for i in range(
            len(edge_classes)
        ):

            for j in range(
                i + 1,
                len(edge_classes),
            ):

                if (
                    i not in atom_indices
                    or j not in atom_indices
                ):

                    continue

                bond_class = int(
                    edge_classes[i, j]
                )

                if bond_class >= len(
                    bond_decoder
                ):

                    continue

                bond_type = bond_decoder[
                    bond_class
                ]

                if bond_type in (
                    None,
                    "",
                    "None",
                ):

                    continue

                try:

                    mol.AddBond(

                        atom_indices[i],

                        atom_indices[j],

                        bond_type,
                    )

                except Exception:

                    continue

        mol = mol.GetMol()

        Chem.SanitizeMol(mol)

        smiles = Chem.MolToSmiles(
            mol,
            canonical=True,
        )

        if not smiles:

            return None, None

        return mol, smiles

    except Exception:

        return None, None


# =====================================================================
# MAIN
# =====================================================================

def main():

    set_seed(
        SEED
    )

    print(
        "=" * 80
    )

    print(
        "ISSUE 3 — CHEMICAL CONDITION EFFECT DIAGNOSTIC"
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
        f"Condition A pIC50: {PIC50_A}"
    )

    print(
        f"Condition B pIC50: {PIC50_B}"
    )

    print(
        f"Both logBB: {LOGBB}"
    )

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
    # LOAD DATASET
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
        f"  Atom features: {nodes}"
    )

    print(
        f"  Bond features: {edges}"
    )

    # =================================================================
    # LOAD GENERATOR
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
    # DATA LOADER
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

    rows = []

    total = 0

    identical_smiles = 0

    different_smiles = 0

    identical_atom_classes = 0

    different_atom_classes = 0

    identical_bond_classes = 0

    different_bond_classes = 0

    total_atom_changes = 0

    total_real_atom_changes = 0

    total_bond_changes = 0

    total_real_bond_changes = 0

    valid_a = 0

    valid_b = 0

    formula_changes = 0

    heavy_atom_changes = 0

    atom_count_changes = 0

    # =================================================================
    # PROCESS BATCHES
    # =================================================================

    for batch_number, batch in enumerate(
        loader
    ):

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
        # SAVE RNG STATE
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
            node_latent_a,
            edge_latent_a,
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
            node_latent_b,
            edge_latent_b,
            node_logits_b,
            edge_logits_b,
        ) = run_generator(

            generator,

            a_tensor,

            x_tensor,

            condition_b_batch,
        )

        # -------------------------------------------------------------
        # CHECK SHAPES
        # -------------------------------------------------------------

        if batch_number == 0:

            print()

            print(
                "Generator output shapes:"
            )

            print(
                f"  node latent A: "
                f"{tuple(node_latent_a.shape)}"
            )

            print(
                f"  edge latent A: "
                f"{tuple(edge_latent_a.shape)}"
            )

            print(
                f"  node logits A: "
                f"{tuple(node_logits_a.shape)}"
            )

            print(
                f"  edge logits A: "
                f"{tuple(edge_logits_a.shape)}"
            )

        # -------------------------------------------------------------
        # ARGMAX
        # -------------------------------------------------------------

        (
            node_classes_a,
            edge_classes_a,
        ) = argmax_graph(

            node_logits_a,

            edge_logits_a,
        )

        (
            node_classes_b,
            edge_classes_b,
        ) = argmax_graph(

            node_logits_b,

            edge_logits_b,
        )

        # -------------------------------------------------------------
        # SAMPLE-BY-SAMPLE
        # -------------------------------------------------------------

        for i in range(
            BATCH_SIZE
        ):

            node_a = (
                node_classes_a[i]
            )

            node_b = (
                node_classes_b[i]
            )

            edge_a = (
                edge_classes_a[i]
            )

            edge_b = (
                edge_classes_b[i]
            )

            # =========================================================
            # ATOM CLASS CHANGES
            # =========================================================

            atom_diff = (
                node_a != node_b
            )

            atom_changes = int(
                atom_diff
                .sum()
                .item()
            )

            # Class 0 is treated as the
            # no-atom / padding class.
            #
            # A change is chemically meaningful
            # if either side contains a real atom.

            real_atom_mask = (

                (node_a != 0)

                |

                (node_b != 0)
            )

            real_atom_changes = int(

                (
                    atom_diff

                    & real_atom_mask
                )

                .sum()

                .item()
            )

            # =========================================================
            # BOND CLASS CHANGES
            # =========================================================

            bond_diff = (
                edge_a != edge_b
            )

            upper_mask = torch.triu(

                torch.ones_like(

                    bond_diff,

                    dtype=torch.bool,
                ),

                diagonal=1,
            )

            unique_bond_diff = (

                bond_diff

                & upper_mask
            )

            bond_changes = int(

                unique_bond_diff

                .sum()

                .item()
            )

            # At least one endpoint must
            # correspond to a real atom.

            real_bond_mask = (

                (
                    node_a
                    .unsqueeze(1)
                    != 0
                )

                |

                (
                    node_b
                    .unsqueeze(1)
                    != 0
                )

                |

                (
                    node_a
                    .unsqueeze(0)
                    != 0
                )

                |

                (
                    node_b
                    .unsqueeze(0)
                    != 0
                )
            )

            real_bond_changes = int(

                (
                    unique_bond_diff

                    & real_bond_mask

                    & upper_mask
                )

                .sum()

                .item()
            )

            # =========================================================
            # ATOM COUNTS
            # =========================================================

            atom_count_a = int(

                (node_a != 0)

                .sum()

                .item()
            )

            atom_count_b = int(

                (node_b != 0)

                .sum()

                .item()
            )

            if (
                atom_count_a
                != atom_count_b
            ):

                atom_count_changes += 1

            # =========================================================
            # RDKit DECODING
            # =========================================================

            mol_a, smiles_a = decode_smiles(

                node_a,

                edge_a,

                atom_decoder,

                bond_decoder,
            )

            mol_b, smiles_b = decode_smiles(

                node_b,

                edge_b,

                atom_decoder,

                bond_decoder,
            )

            if mol_a is not None:

                valid_a += 1

            if mol_b is not None:

                valid_b += 1

            same_smiles = (

                smiles_a is not None

                and smiles_b is not None

                and smiles_a == smiles_b
            )

            if same_smiles:

                identical_smiles += 1

            elif (
                smiles_a is not None
                or smiles_b is not None
            ):

                different_smiles += 1

            # =========================================================
            # FORMULA / HEAVY ATOMS
            # =========================================================

            formula_a = None
            formula_b = None

            heavy_a = None
            heavy_b = None

            if mol_a is not None:

                formula_a = (
                    rdMolDescriptors
                    .CalcMolFormula(
                        mol_a
                    )
                )

                heavy_a = (
                    mol_a
                    .GetNumHeavyAtoms()
                )

            if mol_b is not None:

                formula_b = (
                    rdMolDescriptors
                    .CalcMolFormula(
                        mol_b
                    )
                )

                heavy_b = (
                    mol_b
                    .GetNumHeavyAtoms()
                )

            if (
                formula_a is not None
                and formula_b is not None
                and formula_a != formula_b
            ):

                formula_changes += 1

            if (
                heavy_a is not None
                and heavy_b is not None
                and heavy_a != heavy_b
            ):

                heavy_atom_changes += 1

            # =========================================================
            # GLOBAL COUNTS
            # =========================================================

            if atom_changes == 0:

                identical_atom_classes += 1

            else:

                different_atom_classes += 1

            if bond_changes == 0:

                identical_bond_classes += 1

            else:

                different_bond_classes += 1

            total_atom_changes += (
                atom_changes
            )

            total_real_atom_changes += (
                real_atom_changes
            )

            total_bond_changes += (
                bond_changes
            )

            total_real_bond_changes += (
                real_bond_changes
            )

            # =========================================================
            # SAVE ROW
            # =========================================================

            rows.append({

                "batch":
                    batch_number + 1,

                "sample":
                    i + 1,

                "atom_class_changes":
                    atom_changes,

                "real_atom_class_changes":
                    real_atom_changes,

                "bond_class_changes":
                    bond_changes,

                "real_bond_class_changes":
                    real_bond_changes,

                "atom_count_5_5":
                    atom_count_a,

                "atom_count_8_0":
                    atom_count_b,

                "formula_5_5":
                    formula_a,

                "formula_8_0":
                    formula_b,

                "heavy_atoms_5_5":
                    heavy_a,

                "heavy_atoms_8_0":
                    heavy_b,

                "smiles_5_5":
                    smiles_a,

                "smiles_8_0":
                    smiles_b,

                "same_smiles":
                    same_smiles,
            })

            total += 1

    # =================================================================
    # SUMMARY
    # =================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "CHEMICAL CONDITION EFFECT SUMMARY"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Graph pairs tested: {total}"
    )

    print()

    print(
        "ATOM CHANGES"
    )

    print(
        f"Pairs with any atom-class change: "
        f"{different_atom_classes}/{total}"
    )

    print(
        f"Pairs with no atom-class change: "
        f"{identical_atom_classes}/{total}"
    )

    print(
        f"Total atom-class changes: "
        f"{total_atom_changes}"
    )

    print(
        f"Total chemically meaningful atom changes: "
        f"{total_real_atom_changes}"
    )

    print()

    print(
        "BOND CHANGES"
    )

    print(
        f"Pairs with any bond-class change: "
        f"{different_bond_classes}/{total}"
    )

    print(
        f"Pairs with no bond-class change: "
        f"{identical_bond_classes}/{total}"
    )

    print(
        f"Total bond-class changes: "
        f"{total_bond_changes}"
    )

    print(
        f"Total chemically meaningful bond changes: "
        f"{total_real_bond_changes}"
    )

    print()

    print(
        "MOLECULAR CHANGES"
    )

    print(
        f"Atom-count changes: "
        f"{atom_count_changes}/{total}"
    )

    print(
        f"Formula changes: "
        f"{formula_changes}/{total}"
    )

    print(
        f"Heavy-atom-count changes: "
        f"{heavy_atom_changes}/{total}"
    )

    print()

    print(
        "VALIDITY"
    )

    print(
        f"Valid at pIC50 5.5: "
        f"{valid_a}/{total}"
    )

    print(
        f"Valid at pIC50 8.0: "
        f"{valid_b}/{total}"
    )

    print()

    print(
        "SMILES"
    )

    print(
        f"Identical valid SMILES: "
        f"{identical_smiles}"
    )

    print(
        f"Different valid SMILES: "
        f"{different_smiles}"
    )

    # =================================================================
    # SAVE CSV
    # =================================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()

    print(
        "Detailed results saved:"
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
        "INTERPRETATION"
    )

    print(
        "=" * 80
    )

    if total == 0:

        print(
            "No samples were evaluated."
        )

    elif (
        total_real_atom_changes == 0
        and total_real_bond_changes == 0
    ):

        print(
            "DIAGNOSIS:"
        )

        print(
            "The pIC50 condition changes the graph "
            "representation but does not change "
            "chemically meaningful atoms or bonds."
        )

        print()

        print(
            "The condition signal is therefore being "
            "spent primarily on padding or redundant "
            "graph states."
        )

    elif (
        different_smiles == 0
        and (
            total_real_atom_changes > 0
            or total_real_bond_changes > 0
        )
    ):

        print(
            "DIAGNOSIS:"
        )

        print(
            "Real graph components change, but the "
            "resulting molecules remain chemically "
            "equivalent after RDKit sanitization."
        )

        print()

        print(
            "The next step is to inspect the DrugGEN "
            "categorical graph encoding/decoding."
        )

    elif different_smiles > 0:

        print(
            "DIAGNOSIS:"
        )

        print(
            "The continuous pIC50 condition produces "
            "genuinely different molecules."
        )

        print()

        print(
            "The conditioning mechanism survives "
            "through molecular decoding."
        )

        print()

        print(
            "The next problem is therefore the learned "
            "relationship between generated chemistry "
            "and BACE1 activity."
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
