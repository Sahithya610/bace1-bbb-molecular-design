"""
Generate conditional molecules using the CrossAttentionGenerator checkpoint.

Identical to generate_joint_molecules.py except it instantiates
CrossAttentionGenerator (from train_joint_cross_attention.py) instead of
ConditionalGenerator, so the checkpoint from Issue4 loads with strict=True.
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
from train_joint_cross_attention import CrossAttentionGenerator


# -----------------------------------------------------------------------------
# Project defaults (identical to generate_joint_molecules.py)
# -----------------------------------------------------------------------------
MAX_ATOMS  = 45
DROPOUT    = 0.0
DIM        = 128
DEPTH      = 1
HEADS      = 8
MLP_RATIO  = 3

PIC50_MEAN = 6.691315
PIC50_STD  = 1.308678
BBB_MEAN   = -0.077873
BBB_STD    = 0.751623


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "generator_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def canonical_smiles(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return smiles or None
    except Exception:
        return None


def make_condition(batch_size, desired_pic50, desired_logbb, device):
    normalized_pic50 = (desired_pic50 - PIC50_MEAN) / PIC50_STD
    normalized_pic50 = max(-1.5, min(1.5, normalized_pic50))
    normalized_logbb = (desired_logbb - BBB_MEAN) / BBB_STD
    normalized_logbb = max(-2.0, min(2.0, normalized_logbb))
    print(f"  Condition: pIC50={normalized_pic50:+.4f}  logBB={normalized_logbb:+.4f}")
    row = torch.tensor(
        [1.0, normalized_pic50, normalized_logbb],
        dtype=torch.float32,
        device=device,
    )
    return row.unsqueeze(0).repeat(batch_size, 1)


def build_model(device, vertexes, edges, nodes, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")

    model = CrossAttentionGenerator(
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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(state_dict).__name__}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def largest_component_smiles(smiles):
    if not smiles or "." not in smiles:
        return smiles
    components = [part for part in smiles.split(".") if part]
    if not components:
        return None
    largest = max(components, key=len)
    return canonical_smiles(Chem.MolFromSmiles(largest))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--batch_size",    type=int,   default=128)
    parser.add_argument("--num_batches",   type=int,   default=8)
    parser.add_argument("--desired_pIC50", type=float, default=8.0)
    parser.add_argument("--desired_logBB", type=float, default=0.0)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("CROSS-ATTENTION GENERATOR — molecule generation")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"pIC50={args.desired_pIC50}  logBB={args.desired_logBB}")
    print(f"Batches: {args.num_batches} x {args.batch_size} = {args.num_batches * args.batch_size}")

    chembl_smi = Path("data/chembl_train.smi")
    bace1_smi  = Path("druggen_data/bace1.smi")
    for p in (chembl_smi, bace1_smi):
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    atom_enc, atom_dec, bond_enc, bond_dec = get_encoders_decoders(
        str(chembl_smi), str(bace1_smi), MAX_ATOMS
    )

    dataset = DruggenDataset(
        root="data",
        dataset_file="chembl_train45.pt",
        raw_files=["chembl_train.smi", "pre_filter.pt", "pre_transform.pt"],
        max_atom=MAX_ATOMS,
        features=False,
        atom_encoder=atom_enc,
        atom_decoder=atom_dec,
        bond_encoder=bond_enc,
        bond_decoder=bond_dec,
    )

    vertexes = int(dataset[0].x.shape[0])
    nodes    = int(dataset[0].x.shape[1])
    edges    = len(bond_dec)

    model = build_model(device, vertexes, edges, nodes, args.checkpoint)
    print("Checkpoint loaded.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    all_results = []
    batches_done = 0

    with torch.no_grad():
        for batch_num, batch in enumerate(loader):
            if batches_done >= args.num_batches:
                break

            _, a_tensor, x_tensor = load_molecules(
                data=batch, batch_size=args.batch_size,
                device=device, b_dim=edges, m_dim=nodes,
            )
            condition = make_condition(args.batch_size, args.desired_pIC50, args.desired_logBB, device)
            _, _, node_logits, edge_logits = model(a_tensor, x_tensor, condition)

            generated_nodes = torch.argmax(node_logits, dim=-1)
            generated_edges = torch.argmax(edge_logits, dim=-1)

            for sample_idx, (eg, ng) in enumerate(zip(generated_edges, generated_nodes), start=1):
                node_labels = ng.detach().cpu().numpy()
                edge_labels = eg.detach().cpu().numpy()

                mol = dataset.matrices2mol(node_labels, edge_labels, strict=False)
                smi = canonical_smiles(mol)
                smi = largest_component_smiles(smi)

                all_results.append({
                    "sample_id":    len(all_results) + 1,
                    "smiles":       smi,
                    "valid":        smi is not None,
                    "requested_pIC50": args.desired_pIC50,
                    "requested_logBB": args.desired_logBB,
                    "checkpoint":   str(args.checkpoint),
                    "seed":         args.seed,
                })

            batches_done += 1
            valid_so_far = sum(r["valid"] for r in all_results)
            print(f"Batch {batches_done}/{args.num_batches} | Generated: {len(all_results)} | Valid: {valid_so_far}")

    df = pd.DataFrame(all_results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    total = len(df)
    valid = df["valid"].sum()
    print(f"\nSaved {total} molecules ({valid} valid, {valid/total:.1%}) to {out}")


if __name__ == "__main__":
    main()
