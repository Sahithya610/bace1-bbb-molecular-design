"""Generate conditional DrugGEN molecules for BACE1 activity and BBB targets.

This script uses the existing DrugGEN graph dataset, encoder/decoder mappings,
ConditionalGenerator architecture, and a trained checkpoint. It does not train
or modify model weights. Requested properties are conditioning targets, not
measurements or guarantees about the generated molecules.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.model.conditional_models import ConditionalGenerator


# -----------------------------------------------------------------------------
# Project defaults
# -----------------------------------------------------------------------------
MAX_ATOMS = 45
DROPOUT = 0.0
DIM = 128
DEPTH = 1
HEADS = 8
MLP_RATIO = 3

PIC50_MEAN = 6.691315
PIC50_STD = 1.308678
BBB_MEAN = -0.077873
BBB_STD = 0.751623

DEFAULT_CHECKPOINT = (
    "experiments/models/Issue3-ChemistryReadout/joint_generator.pt"
)
DEFAULT_OUTPUT = "results/decoder_fix_smoke_16.csv"


# -----------------------------------------------------------------------------
# Reproducibility and small helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_state_dict(checkpoint):
    """Support checkpoints saved as a state dict or a wrapped checkpoint."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "generator_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def canonical_smiles(mol):
    """Sanitize and canonicalize a molecule; return None if invalid."""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return smiles or None
    except Exception:
        return None


def make_condition(batch_size, desired_pic50, desired_logbb, device):
    """Build [BACE1 indicator, normalized pIC50, normalized logBB].

    Normalized values are clamped to [-1.5, +1.5] for pIC50 and
    [-2.0, +2.0] for logBB. This matches the clamping applied in
    ConditionalDruggenDataset and prevents extreme condition values
    from saturating atom-position embeddings (which caused all-atom,
    zero-validity graphs when requesting high pIC50 values).
    """
    if PIC50_STD <= 0 or BBB_STD <= 0:
        raise ValueError("Condition normalization standard deviations must be positive.")

    normalized_pic50 = (desired_pic50 - PIC50_MEAN) / PIC50_STD
    normalized_pic50 = max(-1.5, min(1.5, normalized_pic50))   # clamp

    normalized_logbb = (desired_logbb - BBB_MEAN) / BBB_STD
    normalized_logbb = max(-2.0, min(2.0, normalized_logbb))   # clamp

    print(f"  Condition normalized: pIC50={normalized_pic50:+.4f}  logBB={normalized_logbb:+.4f}  (clamped to [-1.5,+1.5] / [-2.0,+2.0])")

    row = torch.tensor(
        [1.0, normalized_pic50, normalized_logbb],
        dtype=torch.float32,
        device=device,
    )
    return row.unsqueeze(0).repeat(batch_size, 1)


def build_model(device, vertexes, edges, nodes, checkpoint_path):
    """Build the configured ConditionalGenerator and load its checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")

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
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = extract_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Unsupported checkpoint object: {type(state_dict).__name__}"
        )

    # Intentionally use strict loading: incompatible architecture/mapping
    # changes must be detected, not silently ignored.
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def largest_component_smiles(smiles):
    """Keep the largest connected component without altering atom identities."""
    if not smiles or "." not in smiles:
        return smiles
    components = [part for part in smiles.split(".") if part]
    if not components:
        return None
    largest = max(components, key=len)
    return canonical_smiles(Chem.MolFromSmiles(largest))


# -----------------------------------------------------------------------------
# Main generation routine
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate conditional DrugGEN molecules for BACE1 activity and BBB. "
            "The requested properties are targets, not guaranteed outcomes."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=2)
    parser.add_argument("--desired_pIC50", type=float, default=8.0)
    parser.add_argument("--desired_logBB", type=float, default=0.0)
    parser.add_argument(
        "--condition_mode",
        choices=["conditional", "zero"],
        default="conditional",
        help=(
            "Condition vector mode: 'conditional' uses [1, norm_pIC50, norm_logBB]; "
            "'zero' uses [0, 0, 0] to match Option A joint-positive discriminator checkpoints."
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than zero")
    if args.num_batches <= 0:
        parser.error("--num_batches must be greater than zero")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("CONDITIONAL DRUGGEN MOLECULE GENERATION")
    print("=" * 72)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Requested pIC50: {args.desired_pIC50}")
    print(f"Requested logBB: {args.desired_logBB}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches: {args.num_batches}")
    print(f"Maximum requested graph count: {args.batch_size * args.num_batches}")
    print(f"Random seed: {args.seed}")
    print()

    chembl_smi = Path("data/chembl_train.smi")
    bace1_smi = Path("druggen_data/bace1.smi")
    dataset_path = Path("data/chembl_train45.pt")
    for required_path in (chembl_smi, bace1_smi, dataset_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required project file not found: {required_path}")

    atom_encoder, atom_decoder, bond_encoder, bond_decoder = get_encoders_decoders(
        str(chembl_smi),
        str(bace1_smi),
        MAX_ATOMS,
    )

    # Check that the atom mapping contains the expected padding and oxygen labels.
    oxygen_classes = [
        int(index)
        for index, atomic_number in atom_decoder.items()
        if int(atomic_number) == 8
    ]
    padding_classes = [
        int(index)
        for index, atomic_number in atom_decoder.items()
        if int(atomic_number) == 0
    ]
    if not oxygen_classes:
        raise RuntimeError("The active atom decoder has no oxygen (atomic number 8).")
    if not padding_classes:
        raise RuntimeError("The active atom decoder has no padding (atomic number 0).")

    print(f"Atom classes: {len(atom_decoder)}")
    print(f"Oxygen class index/indices: {oxygen_classes}")
    print(f"Padding class index/indices: {padding_classes}")
    print(f"Bond classes: {len(bond_decoder)}")

    dataset = DruggenDataset(
        root="data",
        dataset_file="chembl_train45.pt",
        raw_files=[
            "chembl_train.smi",
            "pre_filter.pt",
            "pre_transform.pt",
        ],
        max_atom=MAX_ATOMS,
        features=False,
        atom_encoder=atom_encoder,
        atom_decoder=atom_decoder,
        bond_encoder=bond_encoder,
        bond_decoder=bond_decoder,
    )

    if len(dataset) == 0:
        raise RuntimeError("The ChEMBL graph dataset is empty.")

    vertexes = int(dataset[0].x.shape[0])
    nodes = int(dataset[0].x.shape[1])
    edges = len(bond_decoder)
    if vertexes != MAX_ATOMS:
        print(
            f"WARNING: graph has {vertexes} positions, while MAX_ATOMS={MAX_ATOMS}."
        )

    print(f"General ChEMBL graphs: {len(dataset):,}")
    print(f"Graph dimensions: vertices={vertexes}, node_classes={nodes}, bond_classes={edges}")

    model = build_model(
        device=device,
        vertexes=vertexes,
        edges=edges,
        nodes=nodes,
        checkpoint_path=args.checkpoint,
    )
    print("Conditional checkpoint loaded successfully.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    all_results = []
    batches_generated = 0
    atom_class_counts = {}
    graph_count = 0

    with torch.no_grad():
        for batch_number, batch in enumerate(loader):
            if batches_generated >= args.num_batches:
                break

            actual_batch_size = args.batch_size
            _, a_tensor, x_tensor = load_molecules(
                data=batch,
                batch_size=actual_batch_size,
                device=device,
                b_dim=edges,
                m_dim=nodes,
            )

            if args.condition_mode == "zero":
                condition = torch.zeros(actual_batch_size, 3, device=device)
            else:
                condition = make_condition(
                    actual_batch_size,
                    args.desired_pIC50,
                    args.desired_logBB,
                    device,
                )

            outputs = model(a_tensor, x_tensor, condition)
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
                raise RuntimeError(
                    "Expected ConditionalGenerator forward() to return four outputs "
                    "(_, _, node_logits, edge_logits); check the active model source."
                )

            _, _, node_logits, edge_logits = outputs
            generated_nodes = torch.argmax(node_logits, dim=-1)
            generated_edges = torch.argmax(edge_logits, dim=-1)

            for sample_index, (edge_graph, node_graph) in enumerate(
                zip(generated_edges, generated_nodes), start=1
            ):
                node_labels = node_graph.detach().cpu().numpy()
                edge_labels = edge_graph.detach().cpu().numpy()

                for label in node_labels:
                    label = int(label)
                    atom_class_counts[label] = atom_class_counts.get(label, 0) + 1

                mol = dataset.matrices2mol(
                    node_labels,
                    edge_labels,
                    strict=False,
                )
                smiles = canonical_smiles(mol)
                smiles = largest_component_smiles(smiles)

                atom_count = mol.GetNumAtoms() if mol is not None else None
                oxygen_count = (
                    sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms())
                    if mol is not None
                    else None
                )
                all_results.append(
                    {
                        "sample_id": len(all_results) + 1,
                        "source_batch": batch_number + 1,
                        "source_sample_in_batch": sample_index,
                        "smiles": smiles,
                        "valid": smiles is not None,
                        "decoded_atom_count_before_component_filter": atom_count,
                        "oxygen_count_before_component_filter": oxygen_count,
                        "target": "BACE1",
                        "condition_mode": args.condition_mode,
                        "requested_pIC50": args.desired_pIC50,
                        "requested_logBB": args.desired_logBB,
                        "checkpoint": str(args.checkpoint),
                        "seed": args.seed,
                    }
                )
                graph_count += 1

            batches_generated += 1
            valid_so_far = sum(row["valid"] for row in all_results)
            print(
                f"Batch {batches_generated}/{args.num_batches} | "
                f"Generated: {len(all_results)} | Valid: {valid_so_far}"
            )

    if not all_results:
        raise RuntimeError(
            "No graph batches were generated. Check the dataset and batch size."
        )

    df = pd.DataFrame(all_results)
    valid_df = df.dropna(subset=["smiles"])
    unique_count = valid_df["smiles"].nunique()
    validity = len(valid_df) / len(df) if len(df) else 0.0
    uniqueness = unique_count / len(valid_df) if len(valid_df) else 0.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 72)
    print("GENERATION SUMMARY")
    print("=" * 72)
    print(f"Requested graph count: {args.batch_size * args.num_batches}")
    print(f"Graphs generated: {len(df)}")
    print(f"Valid molecules: {len(valid_df)}")
    print(f"Unique valid molecules: {unique_count}")
    print(f"Validity rate: {validity:.4%}")
    print(f"Uniqueness among valid molecules: {uniqueness:.4%}")
    print(f"Output: {output_path}")

    total_node_positions = sum(atom_class_counts.values())
    print("\nArgmax node-class counts across all graph positions:")
    for class_index in sorted(atom_class_counts):
        atomic_number = atom_decoder.get(class_index)
        label = (
            "PAD"
            if atomic_number == 0
            else Chem.GetPeriodicTable().GetElementSymbol(int(atomic_number))
            if atomic_number is not None
            else "UNKNOWN"
        )
        count = atom_class_counts[class_index]
        percentage = 100.0 * count / total_node_positions if total_node_positions else 0
        print(f"  class {class_index:>2} ({label:>7}): {count:>7} ({percentage:.3f}%)")

    if len(valid_df):
        molecules_with_oxygen = 0
        for smiles in valid_df["smiles"]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and any(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()):
                molecules_with_oxygen += 1
        print(
            "\nValid molecules containing oxygen: "
            f"{molecules_with_oxygen}/{len(valid_df)} "
            f"({molecules_with_oxygen / len(valid_df):.3%})"
        )

    print("\nNote: requested conditions are generation targets, not validated predictions.")
    print("Run the existing property-prediction/evaluation pipeline before claiming joint success.")
    print("=" * 72)


if __name__ == "__main__":
    main()
