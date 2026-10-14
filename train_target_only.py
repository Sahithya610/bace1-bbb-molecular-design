from pathlib import Path
import argparse

import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.model.models import Generator, Discriminator
from src.model.loss import discriminator_loss, generator_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chembl_smi",
        default="data/chembl_train.smi"
    )

    parser.add_argument(
        "--bace1_smi",
        default="druggen_data/bace1.smi"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--output_dir",
        default="experiments/models/DrugGEN-BACE1-TargetOnly"
    )

    args = parser.parse_args()

    # ================================================================
    # DEVICE
    # ================================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("TARGET-ONLY BACE1 DRUGGEN BASELINE")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ================================================================
    # EXPERIMENT CONFIGURATION
    # ================================================================

    max_atom = 45

    dim = 128
    depth = 1
    heads = 8
    mlp_ratio = 3

    dropout = 0
    ddropout = 0

    act = "relu"

    g_lr = 1e-5
    d_lr = 1e-5

    beta1 = 0.9
    beta2 = 0.999

    lambda_gp = 10

    seed = 1

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ================================================================
    # ENCODERS / DECODERS
    # ================================================================

    raw_file = args.chembl_smi
    drug_raw_file = args.bace1_smi

    atom_encoder, atom_decoder, bond_encoder, bond_decoder = (
        get_encoders_decoders(
            raw_file,
            drug_raw_file,
            max_atom
        )
    )

    print(
        f"Atom types: {len(atom_decoder)}"
    )

    print(
        f"Bond types: {len(bond_decoder)}"
    )

    # ================================================================
    # CHEMBL DATASET
    # ================================================================

    print("\nLoading ChEMBL dataset...")

    chembl_dataset = DruggenDataset(
        root="data",
        dataset_file="chembl_train45.pt",
        raw_files=["chembl_train.smi"],
        max_atom=max_atom,
        features=False,
        atom_encoder=atom_encoder,
        atom_decoder=atom_decoder,
        bond_encoder=bond_encoder,
        bond_decoder=bond_decoder,
    )

    print(
        f"ChEMBL graphs: {len(chembl_dataset)}"
    )

    # ================================================================
    # BACE1 DATASET
    # ================================================================

    print("\nLoading BACE1 dataset...")

    bace1_dataset = DruggenDataset(
        root="druggen_data",
        dataset_file="bace145.pt",
        raw_files=["bace1.smi"],
        max_atom=max_atom,
        features=False,
        atom_encoder=atom_encoder,
        atom_decoder=atom_decoder,
        bond_encoder=bond_encoder,
        bond_decoder=bond_decoder,
    )

    print(
        f"BACE1 graphs: {len(bace1_dataset)}"
    )

    # ================================================================
    # DATA LOADERS
    # ================================================================

    # BACE1 is the primary dataset.
    # Approximately 7,682 / 32 = 240 iterations per epoch.

    bace1_loader = DataLoader(
        bace1_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # ChEMBL is sampled alongside BACE1 batches rather than being
    # traversed completely every epoch.
    chembl_loader = DataLoader(
        chembl_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # ================================================================
    # GRAPH DIMENSIONS
    # ================================================================

    m_dim = len(atom_decoder)
    b_dim = len(bond_decoder)

    vertexes = int(
        chembl_dataset[0].x.shape[0]
    )

    print("\nGraph configuration")
    print("-" * 70)

    print(
        f"Maximum atoms: {vertexes}"
    )

    print(
        f"Atom dimension: {m_dim}"
    )

    print(
        f"Bond dimension: {b_dim}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"Epochs: {args.epochs}"
    )

    print(
        f"BACE1 steps/epoch: "
        f"{len(bace1_dataset) // args.batch_size}"
    )

    # ================================================================
    # MODELS
    # ================================================================

    G = Generator(
        act,
        vertexes,
        b_dim,
        m_dim,
        dropout,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_ratio=mlp_ratio,
    ).to(device)

    D = Discriminator(
        act,
        vertexes,
        b_dim,
        m_dim,
        ddropout,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_ratio=mlp_ratio,
    ).to(device)

    print(
        f"\nGenerator parameters: "
        f"{sum(p.numel() for p in G.parameters()):,}"
    )

    print(
        f"Discriminator parameters: "
        f"{sum(p.numel() for p in D.parameters()):,}"
    )

    # ================================================================
    # OPTIMIZERS
    # ================================================================

    optimizer_G = torch.optim.AdamW(
        G.parameters(),
        lr=g_lr,
        betas=(beta1, beta2),
    )

    optimizer_D = torch.optim.AdamW(
        D.parameters(),
        lr=d_lr,
        betas=(beta1, beta2),
    )

    # ================================================================
    # OUTPUT DIRECTORY
    # ================================================================

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ================================================================
    # TRAINING
    # ================================================================

    print("\nStarting target-only training...")
    print("-" * 70)

    for epoch in range(args.epochs):

        G.train()
        D.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        steps = 0

        # ------------------------------------------------------------
        # ChEMBL iterator
        #
        # ChEMBL is much larger than BACE1, so we do NOT iterate
        # through all 1.4 million ChEMBL graphs every epoch.
        # Instead, one ChEMBL batch is sampled for each BACE1 batch.
        # ------------------------------------------------------------

        chembl_iterator = iter(chembl_loader)

        # ------------------------------------------------------------
        # BACE1 is the primary training loop
        # ------------------------------------------------------------

        for bace1_data in bace1_loader:

            # --------------------------------------------------------
            # Get one ChEMBL batch
            # --------------------------------------------------------

            try:
                chembl_data = next(chembl_iterator)

            except StopIteration:
                chembl_iterator = iter(chembl_loader)

                chembl_data = next(chembl_iterator)

            # --------------------------------------------------------
            # Convert ChEMBL batch
            # --------------------------------------------------------

            _, real_a, real_x = load_molecules(
                chembl_data,
                b_dim=b_dim,
                m_dim=m_dim,
                device=device,
                batch_size=args.batch_size,
            )

            # --------------------------------------------------------
            # Convert BACE1 batch
            # --------------------------------------------------------

            _, target_a, target_x = load_molecules(
                bace1_data,
                b_dim=b_dim,
                m_dim=m_dim,
                device=device,
                batch_size=args.batch_size,
            )

            # ========================================================
            # DISCRIMINATOR UPDATE
            # ========================================================

            optimizer_D.zero_grad()

            _, _, d_loss = discriminator_loss(
                G,
                D,
                real_a,
                real_x,
                target_a,
                target_x,
                args.batch_size,
                device,
                lambda_gp,
            )

            d_loss.backward()

            optimizer_D.step()

            # ========================================================
            # GENERATOR UPDATE
            # ========================================================

            optimizer_G.zero_grad()

            (
                g_loss,
                _,
                _,
                _,
                _
            ) = generator_loss(
                G,
                D,
                target_a,
                target_x,
                args.batch_size,
            )

            g_loss.backward()

            optimizer_G.step()

            # ========================================================
            # RECORD LOSSES
            # ========================================================

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            steps += 1

        # ============================================================
        # EPOCH RESULTS
        # ============================================================

        avg_g = (
            epoch_g_loss / steps
        )

        avg_d = (
            epoch_d_loss / steps
        )

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"G={avg_g:.4f} | "
            f"D={avg_d:.4f}"
        )

        # ============================================================
        # CHECKPOINTS
        # ============================================================

        torch.save(
            G.state_dict(),
            output_dir /
            f"generator_epoch_{epoch + 1:03d}.pt"
        )

        torch.save(
            D.state_dict(),
            output_dir /
            f"discriminator_epoch_{epoch + 1:03d}.pt"
        )

    # ================================================================
    # FINAL MODELS
    # ================================================================

    torch.save(
        G.state_dict(),
        output_dir /
        "target_only_generator.pt"
    )

    torch.save(
        D.state_dict(),
        output_dir /
        "target_only_discriminator.pt"
    )

    # ================================================================
    # COMPLETE
    # ================================================================

    print("\n" + "=" * 70)
    print("TARGET-ONLY TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Generator:     "
        f"{output_dir / 'target_only_generator.pt'}"
    )

    print(
        f"Discriminator: "
        f"{output_dir / 'target_only_discriminator.pt'}"
    )


if __name__ == "__main__":
    main()