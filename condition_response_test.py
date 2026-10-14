"""
Condition dose-response test for the conditional DrugGEN generator.

Question answered (handoff section 64):

    Does increasing the REQUESTED pIC50 (and logBB) actually shift the
    PREDICTED pIC50 (and logBB) of the generated molecules?

Design
------
* The generator input is the general ChEMBL graph distribution
  (data/chembl_train45.pt), exactly as in training (Issue 1 fix).
* A fixed set of ChEMBL input batches is drawn ONCE and re-used for every
  requested condition. Therefore any difference between conditions is caused
  by the condition vector alone, not by different random inputs.
* Generated graphs are decoded exactly like generate_joint_molecules.py
  (argmax -> matrices2mol -> sanitize -> largest fragment -> '*' -> 'C').
* Properties are PREDICTOR outputs (BACE1 ExtraTrees, BBB ExtraTrees), never
  experimental values. Thresholds are NOT changed:
      BACE1 : pIC50 >= 7.6576
      BBB   : logBB >= 0

Sweeps
------
  --pic50_values  requested pIC50 values, evaluated at --fixed_logbb
  --logbb_values  (optional) requested logBB values, evaluated at --fixed_pic50

Example
-------
python condition_response_test.py `
    --checkpoint experiments\\models\\DrugGEN-BACE1-AffinityBBB-Conditioned-v3\\joint_generator.pt `
    --output_dir results\\condition_response_v3
"""

import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

RDLogger.DisableLog("rdApp.*")

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.model.conditional_models import ConditionalGenerator


# ============================================================
# Constants (identical to training / generation / evaluation)
# ============================================================

PIC50_MEAN = 6.691315
PIC50_STD = 1.308678

BBB_MEAN = -0.077873
BBB_STD = 0.751623

BACE1_THRESHOLD = 7.6576
BBB_THRESHOLD = 0.0

FP_RADIUS = 2
FP_BITS = 2048


# ============================================================
# Helpers
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_smiles(mol):
    """Sanitize and canonicalize; None if impossible."""

    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def decode_graph(dataset, node_labels, edge_labels):
    """
    Same post-processing as generate_joint_molecules.py.
    """

    mol = dataset.matrices2mol(
        node_labels,
        edge_labels,
        strict=False,
    )

    smiles = canonical_smiles(mol)

    if smiles is not None and "." in smiles:
        smiles = max(smiles.split("."), key=len)
        smiles = canonical_smiles(Chem.MolFromSmiles(smiles))

    if smiles is not None and "*" in smiles:
        smiles = smiles.replace("*", "C")
        smiles = canonical_smiles(Chem.MolFromSmiles(smiles))

    return smiles


def fingerprint(smiles):
    """Same fingerprint as predict_joint.py (Morgan r=2, 2048 bits)."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=FP_RADIUS,
        nBits=FP_BITS,
    )

    return np.asarray(fp, dtype=np.float32)


def build_generator(device, vertexes, edges, nodes, checkpoint):
    model = ConditionalGenerator(
        act="relu",
        vertexes=vertexes,
        edges=edges,
        nodes=nodes,
        dropout=0,
        dim=128,
        depth=1,
        heads=8,
        mlp_ratio=3,
        condition_dim=3,
    ).to(device)

    state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    return model


def parse_args():
    p = argparse.ArgumentParser(
        description="Condition dose-response test for conditional DrugGEN."
    )

    p.add_argument("--checkpoint", required=True)

    p.add_argument("--chembl_smi", default="data/chembl_train.smi")
    p.add_argument("--bace1_smi", default="druggen_data/bace1.smi")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--max_atom", type=int, default=45)

    p.add_argument(
        "--bace1_model",
        default="results/bace1_activity_model/bace1_activity_model.joblib",
    )
    p.add_argument(
        "--bbb_model",
        default="results/bbb_activity_model/bbb_model.joblib",
    )

    p.add_argument(
        "--pic50_values",
        type=float,
        nargs="+",
        default=[5.5, 6.5, 7.0, 8.0, 9.0],
        help="Requested pIC50 values (evaluated at --fixed_logbb).",
    )
    p.add_argument("--fixed_logbb", type=float, default=0.0)

    p.add_argument(
        "--logbb_values",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Optional: requested logBB values "
            "(evaluated at --fixed_pic50)."
        ),
    )
    p.add_argument("--fixed_pic50", type=float, default=8.0)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--output_dir",
        default="results/condition_response",
    )

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONDITION DOSE-RESPONSE TEST")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    # --------------------------------------------------------
    # Encoders / general ChEMBL dataset (generator INPUT)
    # --------------------------------------------------------
    atom_enc, atom_dec, bond_enc, bond_dec = get_encoders_decoders(
        args.chembl_smi,
        args.bace1_smi,
        args.max_atom,
    )

    chembl_name = (
        Path(args.chembl_smi).stem + str(args.max_atom) + ".pt"
    )

    general = DruggenDataset(
        args.data_dir,
        chembl_name,
        args.chembl_smi,
        args.max_atom,
        False,
        atom_enc,
        atom_dec,
        bond_enc,
        bond_dec,
    )

    m_dim = len(atom_dec)
    b_dim = len(bond_dec)
    vertexes = int(general[0].x.shape[0])

    print(f"ChEMBL input graphs : {len(general):,}")
    print(f"Vertices / atoms / bonds : {vertexes} / {m_dim} / {b_dim}")

    generator = build_generator(
        device, vertexes, b_dim, m_dim, args.checkpoint
    )

    # --------------------------------------------------------
    # Draw the fixed input batches ONCE (shared by all conditions)
    # --------------------------------------------------------
    loader = DataLoader(
        general,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    cached = []

    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break

        _, adj, x = load_molecules(
            batch,
            b_dim=b_dim,
            m_dim=m_dim,
            device=device,
            batch_size=args.batch_size,
        )

        cached.append((adj, x))

    n_inputs = len(cached) * args.batch_size

    print(f"Fixed input graphs shared by every condition: {n_inputs}")

    # --------------------------------------------------------
    # Conditions to test
    # --------------------------------------------------------
    conditions = [
        (p, args.fixed_logbb, "pIC50-sweep")
        for p in args.pic50_values
    ] + [
        (args.fixed_pic50, b, "logBB-sweep")
        for b in args.logbb_values
    ]

    # --------------------------------------------------------
    # Predictors
    # --------------------------------------------------------
    bace1_model = joblib.load(args.bace1_model)
    bbb_model = joblib.load(args.bbb_model)

    summary_rows = []
    all_rows = []

    for pic50, logbb, sweep in conditions:

        cond_vec = torch.tensor(
            [
                1.0,
                (pic50 - PIC50_MEAN) / PIC50_STD,
                (logbb - BBB_MEAN) / BBB_STD,
            ],
            dtype=torch.float32,
            device=device,
        )

        smiles_all = []

        for adj, x in cached:

            condition = cond_vec.unsqueeze(0).repeat(adj.shape[0], 1)

            with torch.inference_mode():
                _, _, node_logits, edge_logits = generator(
                    adj, x, condition
                )

            nodes = node_logits.argmax(dim=-1).cpu().numpy()
            edges = edge_logits.argmax(dim=-1).cpu().numpy()

            for n_lab, e_lab in zip(nodes, edges):
                smiles_all.append(
                    decode_graph(general, n_lab, e_lab)
                )

        total = len(smiles_all)
        valid = [s for s in smiles_all if s is not None]
        unique = list(dict.fromkeys(valid))

        row = {
            "sweep": sweep,
            "requested_pIC50": pic50,
            "requested_logBB": logbb,
            "generated": total,
            "valid": len(valid),
            "validity": len(valid) / total if total else 0.0,
            "unique_valid": len(unique),
            "uniqueness": len(unique) / len(valid) if valid else 0.0,
        }

        # ----------------------------------------------------
        # Predict on unique valid molecules
        # ----------------------------------------------------
        fps, kept = [], []

        for s in unique:
            fp = fingerprint(s)
            if fp is not None:
                fps.append(fp)
                kept.append(s)

        if kept:
            X = np.vstack(fps).astype(np.float32)
            p_pic50 = bace1_model.predict(X)
            p_logbb = bbb_model.predict(X)

            bace_ok = p_pic50 >= BACE1_THRESHOLD
            bbb_ok = p_logbb >= BBB_THRESHOLD

            row.update(
                {
                    "pIC50_mean": float(np.mean(p_pic50)),
                    "pIC50_median": float(np.median(p_pic50)),
                    "pIC50_p90": float(np.percentile(p_pic50, 90)),
                    "pIC50_max": float(np.max(p_pic50)),
                    "pct_pIC50_ge_7.6576": 100.0 * float(np.mean(bace_ok)),
                    "pct_pIC50_ge_8.0": 100.0 * float(np.mean(p_pic50 >= 8.0)),
                    "logBB_mean": float(np.mean(p_logbb)),
                    "logBB_median": float(np.median(p_logbb)),
                    "pct_logBB_ge_0": 100.0 * float(np.mean(bbb_ok)),
                    "pct_joint": 100.0 * float(np.mean(bace_ok & bbb_ok)),
                    "n_joint": int(np.sum(bace_ok & bbb_ok)),
                }
            )

            for s, a, b in zip(kept, p_pic50, p_logbb):
                all_rows.append(
                    {
                        "smiles": s,
                        "sweep": sweep,
                        "desired_pIC50": pic50,
                        "desired_logBB": logbb,
                        "predicted_pIC50": float(a),
                        "predicted_logBB": float(b),
                    }
                )

            pd.DataFrame(
                {
                    "smiles": kept,
                    "target": "BACE1",
                    "desired_pIC50": pic50,
                    "desired_logBB": logbb,
                    "predicted_pIC50": p_pic50,
                    "predicted_logBB": p_logbb,
                }
            ).to_csv(
                out_dir / f"generated_pIC50_{pic50}_logBB_{logbb}.csv",
                index=False,
            )

        summary_rows.append(row)

        print(
            f"[{sweep}] requested pIC50={pic50:<5} logBB={logbb:<5} | "
            f"valid={row['valid']}/{total} unique={row['unique_valid']} | "
            f"pIC50 mean={row.get('pIC50_mean', float('nan')):.3f} "
            f"max={row.get('pIC50_max', float('nan')):.3f} | "
            f"logBB mean={row.get('logBB_mean', float('nan')):.3f} | "
            f"joint={row.get('n_joint', 0)}"
        )

    # --------------------------------------------------------
    # Save + summarise
    # --------------------------------------------------------
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "condition_response_summary.csv", index=False)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            out_dir / "condition_response_all_molecules.csv",
            index=False,
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    cols = [
        "sweep", "requested_pIC50", "requested_logBB", "validity",
        "unique_valid", "pIC50_mean", "pIC50_max",
        "pct_pIC50_ge_7.6576", "pct_pIC50_ge_8.0",
        "logBB_mean", "pct_logBB_ge_0", "pct_joint",
    ]

    print(
        summary[[c for c in cols if c in summary.columns]]
        .round(3)
        .to_string(index=False)
    )

    # ---------------- pIC50 dose-response ----------------
    pic = pd.DataFrame(all_rows)

    if not pic.empty:
        pic = pic[pic["sweep"] == "pIC50-sweep"]

    print()
    if len(pic) > 2 and pic["desired_pIC50"].nunique() > 1:

        lo, hi = min(args.pic50_values), max(args.pic50_values)

        m_lo = pic.loc[pic["desired_pIC50"] == lo, "predicted_pIC50"].mean()
        m_hi = pic.loc[pic["desired_pIC50"] == hi, "predicted_pIC50"].mean()

        rho, pval = spearmanr(
            pic["desired_pIC50"], pic["predicted_pIC50"]
        )

        print("pIC50 dose-response (identical input graphs for every condition)")
        print("-" * 70)
        print(f"Mean predicted pIC50 at requested {lo}: {m_lo:.3f}")
        print(f"Mean predicted pIC50 at requested {hi}: {m_hi:.3f}")
        print(f"Shift (high - low)                     : {m_hi - m_lo:+.3f}")
        print(f"Spearman(requested, predicted)         : {rho:+.3f} (p={pval:.2g})")
    else:
        print("Not enough valid molecules for a pIC50 dose-response estimate.")

    # ---------------- logBB dose-response ----------------
    bb = pd.DataFrame(all_rows)

    if not bb.empty:
        bb = bb[bb["sweep"] == "logBB-sweep"]

    if len(bb) > 2 and bb["desired_logBB"].nunique() > 1:

        rho_b, p_b = spearmanr(
            bb["desired_logBB"], bb["predicted_logBB"]
        )

        print()
        print("logBB dose-response")
        print("-" * 70)
        print(f"Spearman(requested, predicted)         : {rho_b:+.3f} (p={p_b:.2g})")

    print()
    print("All values are PREDICTOR outputs, not experimental measurements.")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
