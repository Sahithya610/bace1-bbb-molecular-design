import torch
import numpy as np

from torch_geometric.loader import DataLoader

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.model.conditional_models import ConditionalGenerator


PIC50_MEAN = 6.691315
PIC50_STD = 1.308678

BBB_MEAN = -0.077873
BBB_STD = 0.751623


CHECKPOINT = (
    "experiments/models/"
    "DrugGEN-BACE1-AffinityBBB-Conditioned/"
    "joint_generator.pt"
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("CONDITIONAL GENERATOR OUTPUT DIAGNOSTIC")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load encoders / decoders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load BACE1 dataset
    # --------------------------------------------------------

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

    batch_size = 32

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    mol_batch = next(iter(loader))

    vertices = int(dataset[0].x.shape[0])
    nodes = int(dataset[0].x.shape[1])
    edges = len(bond_decoder)

    # --------------------------------------------------------
    # Convert graphs to DrugGEN tensors
    # --------------------------------------------------------

    _, a_tensor, x_tensor = load_molecules(
        data=mol_batch,
        batch_size=batch_size,
        device=device,
        b_dim=edges,
        m_dim=nodes,
    )

    # --------------------------------------------------------
    # Mean training condition
    # --------------------------------------------------------

    desired_pIC50 = PIC50_MEAN
    desired_logBB = BBB_MEAN

    normalized_pIC50 = (
        desired_pIC50 - PIC50_MEAN
    ) / PIC50_STD

    normalized_logBB = (
        desired_logBB - BBB_MEAN
    ) / BBB_STD

    condition = torch.tensor(
        [
            1.0,
            normalized_pIC50,
            normalized_logBB,
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    condition = condition.repeat(
        a_tensor.shape[0],
        1,
    )

    print()
    print("Condition:")
    print(
        "[BACE1=1.0, "
        f"pIC50_norm={normalized_pIC50:.4f}, "
        f"logBB_norm={normalized_logBB:.4f}]"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = ConditionalGenerator(
        act="relu",
        vertexes=vertices,
        edges=edges,
        nodes=nodes,
        dropout=0,
        dim=128,
        depth=1,
        heads=8,
        mlp_ratio=3,
        condition_dim=3,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # --------------------------------------------------------
    # Generate logits
    # --------------------------------------------------------

    with torch.inference_mode():

        _, _, node_logits, edge_logits = model(
            a_tensor,
            x_tensor,
            condition,
        )

    print()
    print("-" * 70)
    print("RAW OUTPUT SHAPES")
    print("-" * 70)

    print(
        "Node logits:",
        tuple(node_logits.shape)
    )

    print(
        "Edge logits:",
        tuple(edge_logits.shape)
    )

    # --------------------------------------------------------
    # Argmax classes
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
    # Node class distribution
    # --------------------------------------------------------

    node_counts = torch.bincount(
        generated_nodes.reshape(-1),
        minlength=nodes,
    )

    print()
    print("-" * 70)
    print("NODE CLASS DISTRIBUTION")
    print("-" * 70)

    for i, count in enumerate(
        node_counts.tolist()
    ):

        percentage = (
            100.0 * count
            / generated_nodes.numel()
        )

        label = atom_decoder.get(
            i,
            str(i)
        )

        print(
            f"class {i:2d} | "
            f"decoder={label!s:>8} | "
            f"count={count:6d} | "
            f"{percentage:7.2f}%"
        )

    # --------------------------------------------------------
    # Edge class distribution
    # --------------------------------------------------------

    edge_counts = torch.bincount(
        generated_edges.reshape(-1),
        minlength=edges,
    )

    print()
    print("-" * 70)
    print("EDGE CLASS DISTRIBUTION")
    print("-" * 70)

    for i, count in enumerate(
        edge_counts.tolist()
    ):

        percentage = (
            100.0 * count
            / generated_edges.numel()
        )

        label = bond_decoder.get(
            i,
            str(i)
        )

        print(
            f"class {i:2d} | "
            f"decoder={label!s:>8} | "
            f"count={count:8d} | "
            f"{percentage:7.2f}%"
        )

    # --------------------------------------------------------
    # Graph-level statistics
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("GRAPH-LEVEL NODE STATISTICS")
    print("-" * 70)

    unique_counts = []

    for graph_nodes in generated_nodes:

        unique, counts = torch.unique(
            graph_nodes,
            return_counts=True,
        )

        # Print number of non-zero / non-padding classes.
        nonzero = int(
            (graph_nodes != 0).sum().item()
        )

        unique_counts.append(
            len(unique)
        )

        print(
            f"nonzero node positions: "
            f"{nonzero:2d} | "
            f"unique node classes: "
            f"{len(unique):2d}"
        )

    print()
    print(
        "Average unique node classes:",
        np.mean(unique_counts)
    )

    # --------------------------------------------------------
    # Logit statistics
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOGIT STATISTICS")
    print("-" * 70)

    print(
        "Node logits mean:",
        float(node_logits.mean())
    )

    print(
        "Node logits std:",
        float(node_logits.std())
    )

    print(
        "Node logits min:",
        float(node_logits.min())
    )

    print(
        "Node logits max:",
        float(node_logits.max())
    )

    print()

    print(
        "Edge logits mean:",
        float(edge_logits.mean())
    )

    print(
        "Edge logits std:",
        float(edge_logits.std())
    )

    print(
        "Edge logits min:",
        float(edge_logits.min())
    )

    print(
        "Edge logits max:",
        float(edge_logits.max())
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()