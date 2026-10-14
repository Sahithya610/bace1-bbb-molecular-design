"""
DrugGEN Target-Only BACE1 Molecule Generation

Purpose:
    Generate molecules using the trained target-only DrugGEN model.

Target:
    BACE1

Model:
    Original DrugGEN Generator
    trained with BACE1 target molecules

Generation mechanism:
    BACE1 graph tensors
        -> DrugGEN Generator
        -> node/edge logits
        -> argmax graph labels
        -> matrices2mol()
        -> RDKit molecule
        -> canonical SMILES

Important:
    This script does NOT use BBB conditioning.
    It is the generation stage for the target-only baseline.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from torch_geometric.loader import DataLoader

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.model.models import Generator


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SMILES conversion
# ============================================================

def canonical_smiles(mol):
    """
    Convert a valid RDKit molecule into canonical SMILES.

    Returns None if the molecule cannot be sanitized.
    """

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


# ============================================================
# Load target-only DrugGEN generator
# ============================================================

def build_model(
    device,
    vertexes,
    edges,
    nodes,
    checkpoint,
):
    """
    Reconstruct the original DrugGEN Generator architecture
    used by the target-only training script.
    """

    model = Generator(
        act="relu",
        vertexes=vertexes,
        edges=edges,
        nodes=nodes,
        dropout=0,
        dim=128,
        depth=1,
        heads=8,
        mlp_ratio=3,
    ).to(device)

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    # Support either a plain state_dict or checkpoint dictionary.
    if (
        isinstance(checkpoint_data, dict)
        and "model_state_dict" in checkpoint_data
    ):
        state_dict = checkpoint_data["model_state_dict"]

    elif (
        isinstance(checkpoint_data, dict)
        and "state_dict" in checkpoint_data
    ):
        state_dict = checkpoint_data["state_dict"]

    else:
        state_dict = checkpoint_data

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ============================================================
# Generate one batch
# ============================================================

def generate_batch(
    model,
    dataset,
    batch_size,
    device,
):
    """
    Generate one batch using the original DrugGEN inference
    mechanism.

    Real BACE1 graph tensors are used as the generator input,
    matching the DrugGEN inference mechanism used previously.

    No BBB condition is applied.
    """

    model.eval()

    vertices = int(
        dataset[0].x.shape[0]
    )

    nodes = int(
        dataset[0].x.shape[1]
    )

    edges = len(
        dataset.bond_decoder_m
    )

    # --------------------------------------------------------
    # Select a real BACE1 graph batch as generator input
    # --------------------------------------------------------

    inference_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    mol_batch = next(
        iter(inference_loader)
    )

    # --------------------------------------------------------
    # Convert BACE1 graph batch into DrugGEN tensors
    # --------------------------------------------------------

    _, a_tensor, x_tensor = load_molecules(
        data=mol_batch,
        batch_size=batch_size,
        device=device,
        b_dim=edges,
        m_dim=nodes,
    )

    # --------------------------------------------------------
    # Original target-only DrugGEN generation
    # --------------------------------------------------------

    with torch.inference_mode():

        _, _, node_logits, edge_logits = model(
            a_tensor,
            x_tensor,
        )

    # --------------------------------------------------------
    # Convert logits into discrete atom/bond classes
    # --------------------------------------------------------

    generated_nodes = torch.argmax(
        node_logits,
        dim=-1,
    )

    generated_edges = torch.argmax(
        edge_logits,
        dim=-1,
    )

    # --------------------------------------------------------
    # Graph -> RDKit molecule -> canonical SMILES
    # --------------------------------------------------------

    batch_results = []

    for edge_graph, node_graph in zip(
        generated_edges,
        generated_nodes,
    ):

        mol = dataset.matrices2mol(
            node_graph.detach().cpu().numpy(),
            edge_graph.detach().cpu().numpy(),
            strict=False,
        )

        smiles = canonical_smiles(mol)

        # ----------------------------------------------------
        # Keep largest connected component if necessary
        # ----------------------------------------------------

        if smiles is not None and "." in smiles:

            smiles = max(
                smiles.split("."),
                key=len,
            )

            mol = Chem.MolFromSmiles(
                smiles
            )

            smiles = canonical_smiles(
                mol
            )

        # ----------------------------------------------------
        # Preserve original DrugGEN wildcard handling
        # ----------------------------------------------------

        if smiles is not None and "*" in smiles:

            smiles = smiles.replace(
                "*",
                "C",
            )

            mol = Chem.MolFromSmiles(
                smiles
            )

            smiles = canonical_smiles(
                mol
            )

        batch_results.append(
            {
                "smiles": smiles,
                "target": "BACE1",
            }
        )

    return batch_results


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/models/"
            "DrugGEN-BACE1-TargetOnly/"
            "target_only_generator.pt"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/"
            "target_only_generated_1024.csv"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # ========================================================
    # Reproducibility
    # ========================================================

    set_seed(args.seed)

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("DrugGEN TARGET-ONLY BACE1 Generation")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ========================================================
    # Load existing DrugGEN encoders/decoders
    # ========================================================

    print()
    print(
        "Loading existing DrugGEN "
        "encoders/decoders..."
    )

    (
        atom_encoder,
        atom_decoder,
        bond_encoder,
        bond_decoder,
    ) = get_encoders_decoders(
        "data/chembl_train.smi",
        "druggen_data/bace1.smi",
        45,
    )

    print(
        f"Atom types: "
        f"{len(atom_decoder)}"
    )

    print(
        f"Bond types: "
        f"{len(bond_decoder)}"
    )

    # ========================================================
    # Load processed BACE1 graph dataset
    # ========================================================

    print()
    print(
        "Loading processed BACE1 dataset..."
    )

    dataset = DruggenDataset(
        root="druggen_data",
        dataset_file="bace145.pt",
        raw_files=[
            "bace1.smi",
            "pre_filter.pt",
            "pre_transform.pt",
        ],
        max_atom=45,
        features=False,
        atom_encoder=atom_encoder,
        atom_decoder=atom_decoder,
        bond_encoder=bond_encoder,
        bond_decoder=bond_decoder,
    )

    print(
        f"BACE1 graph count: "
        f"{len(dataset)}"
    )

    print(
        f"Node tensor shape: "
        f"{dataset[0].x.shape}"
    )

    print(
        f"Bond tensor shape: "
        f"{dataset[0].edge_index.shape}"
    )

    # ========================================================
    # Determine graph dimensions
    # ========================================================

    vertexes = int(
        dataset[0].x.shape[0]
    )

    nodes = int(
        dataset[0].x.shape[1]
    )

    edges = len(
        bond_decoder
    )

    print(
        f"Vertices: {vertexes}"
    )

    print(
        f"Node classes: {nodes}"
    )

    print(
        f"Edge classes: {edges}"
    )

    # ========================================================
    # Load trained target-only generator
    # ========================================================

    print()
    print(
        "Loading target-only generator..."
    )

    print(
        f"Checkpoint: "
        f"{args.checkpoint}"
    )

    model = build_model(
        device=device,
        vertexes=vertexes,
        edges=edges,
        nodes=nodes,
        checkpoint=args.checkpoint,
    )

    print(
        "Target-only generator "
        "loaded successfully."
    )

    # ========================================================
    # Generate molecules
    # ========================================================

    all_results = []

    for batch_number in range(
        args.num_batches
    ):

        print()
        print(
            f"Generating batch "
            f"{batch_number + 1}/"
            f"{args.num_batches}..."
        )

        batch_results = generate_batch(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
        )

        all_results.extend(
            batch_results
        )

        valid = sum(
            result["smiles"] is not None
            for result in batch_results
        )

        print(
            f"Valid molecules: "
            f"{valid}/"
            f"{len(batch_results)}"
        )

    # ========================================================
    # Basic generation evaluation
    # ========================================================

    df = pd.DataFrame(
        all_results
    )

    total = len(df)

    valid_df = df.dropna(
        subset=["smiles"]
    ).copy()

    unique_valid = (
        valid_df["smiles"]
        .nunique()
    )

    if total > 0:

        validity = (
            len(valid_df)
            / total
        )

    else:

        validity = 0.0

    if len(valid_df) > 0:

        uniqueness = (
            unique_valid
            / len(valid_df)
        )

    else:

        uniqueness = 0.0

    # ========================================================
    # Save results
    # ========================================================

    output_directory = os.path.dirname(
        args.output
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    df.to_csv(
        args.output,
        index=False,
    )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 70)
    print("TARGET-ONLY GENERATION SUMMARY")
    print("=" * 70)

    print(
        f"Total generated graphs: "
        f"{total}"
    )

    print(
        f"Valid molecules: "
        f"{len(valid_df)}"
    )

    print(
        f"Unique valid molecules: "
        f"{unique_valid}"
    )

    print(
        f"Validity: "
        f"{validity:.4f}"
    )

    print(
        f"Uniqueness: "
        f"{uniqueness:.4f}"
    )

    print(
        f"Output: "
        f"{args.output}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()