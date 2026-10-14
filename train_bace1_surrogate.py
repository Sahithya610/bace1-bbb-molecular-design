"""
Train a differentiable MLP surrogate of the BACE1 ExtraTrees pIC50 predictor.

The surrogate is trained on the same 8,271 molecules and Morgan ECFP4
fingerprints (radius=2, 2048 bits) used to train the ExtraTrees oracle,
using the same scaffold split.

Purpose
-------
The ExtraTrees model cannot receive gradients.  A small MLP trained to
approximate it can, enabling a generator loss term that directly penalises
the Morgan-fingerprint-based pIC50 prediction of generated molecules.

The surrogate does NOT need to be more accurate than ExtraTrees.  It only
needs to be:

  1. Differentiable with respect to its Morgan-fingerprint input.
  2. Correlated enough with ExtraTrees that optimising the surrogate moves
     the ExtraTrees score in the same direction.

Target validation: scaffold-test R² ≥ 0.50 (ExtraTrees achieves 0.616).
If R² < 0.40, retrain with a deeper/wider MLP before using it in the
generator loop.

Outputs
-------
results/bace1_surrogate_model/
    bace1_surrogate.pt          PyTorch MLP state dict
    surrogate_metadata.json     architecture + test metrics

Usage
-----
    .venv/Scripts/python.exe train_bace1_surrogate.py
        --input druggen_data/bace1_continuous_bbb.csv
        --output_dir results/bace1_surrogate_model

"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================
# Fingerprint constants — must match predict_joint.py and the
# ExtraTrees training script (train_bace1_activity.py).
# ============================================================

RADIUS = 2
N_BITS = 2048
RANDOM_STATE = 42

# ============================================================
# Normalization constants — same as the generator.
# ============================================================

PIC50_MEAN = 6.691315
PIC50_STD = 1.308678


# ============================================================
# Utilities
# ============================================================

def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=RADIUS, nBits=N_BITS)
    return np.asarray(fp, dtype=np.float32)


def scaffold_split(df):
    """Deterministic scaffold split matching train_bace1_activity.py exactly."""
    groups = {}
    for i, scaf in enumerate(df["scaffold"]):
        groups.setdefault(scaf, []).append(i)

    rng = np.random.RandomState(RANDOM_STATE)
    scaffold_ids = list(groups.keys())
    rng.shuffle(scaffold_ids)

    train_target = int(0.70 * len(df))
    val_target = int(0.15 * len(df))

    train_idx, val_idx, test_idx = [], [], []
    for scaf in scaffold_ids:
        members = groups[scaf]
        if len(train_idx) < train_target:
            train_idx.extend(members)
        elif len(val_idx) < val_target:
            val_idx.extend(members)
        else:
            test_idx.extend(members)

    return (
        np.array(train_idx, dtype=int),
        np.array(val_idx, dtype=int),
        np.array(test_idx, dtype=int),
    )


# ============================================================
# MLP surrogate
# ============================================================

class BACE1SurrogateMLP(nn.Module):
    """
    Small MLP that maps Morgan ECFP4 fingerprint (2048-dim binary vector)
    to normalized pIC50.

    Architecture: 2048 -> 512 -> 256 -> 128 -> 1

    The output is in normalized pIC50 space
    (mean=6.691315, std=1.308678).

    To obtain raw pIC50: raw = out * PIC50_STD + PIC50_MEAN.

    Input contract:
        x: float tensor of shape [B, 2048], values in {0.0, 1.0}
           (or soft counts in [0.0, 1.0] during generator training).
    """

    def __init__(self, n_bits: int = N_BITS, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 1] normalized pIC50."""
        return self.net(x)


# ============================================================
# Training
# ============================================================

def train(cfg):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load and preprocess data
    # --------------------------------------------------------
    print("Loading data ...")
    df = pd.read_csv(cfg.input)
    df = df[["canonical_smiles", "pIC50"]].copy()
    df["canonical_smiles"] = df["canonical_smiles"].map(canonicalize)
    df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    df = df.dropna(subset=["canonical_smiles", "pIC50"])

    # Median pIC50 per unique molecule (same as ExtraTrees training).
    df = df.groupby("canonical_smiles", as_index=False)["pIC50"].median()
    df["scaffold"] = df["canonical_smiles"].map(scaffold)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)

    print(f"Unique molecules : {len(df)}")

    fps, keep = [], []
    for i, smi in enumerate(df["canonical_smiles"]):
        fp = fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            keep.append(i)

    df = df.iloc[keep].reset_index(drop=True)
    X = np.vstack(fps).astype(np.float32)             # [N, 2048]
    y_raw = df["pIC50"].to_numpy(dtype=np.float32)     # raw pIC50

    # Normalize targets to the same space the generator uses.
    y_norm = (y_raw - PIC50_MEAN) / PIC50_STD

    # --------------------------------------------------------
    # Scaffold split (identical to ExtraTrees training)
    # --------------------------------------------------------
    train_idx, val_idx, test_idx = scaffold_split(df)
    print(
        f"Split: train={len(train_idx)} / val={len(val_idx)} / test={len(test_idx)}"
    )

    X_tr = torch.from_numpy(X[train_idx])
    y_tr = torch.from_numpy(y_norm[train_idx]).unsqueeze(1)
    X_va = torch.from_numpy(X[val_idx])
    y_va = torch.from_numpy(y_norm[val_idx]).unsqueeze(1)
    X_te = torch.from_numpy(X[test_idx])
    y_te = torch.from_numpy(y_norm[test_idx]).unsqueeze(1)

    # --------------------------------------------------------
    # Model, optimizer, scheduler
    # --------------------------------------------------------
    model = BACE1SurrogateMLP(n_bits=N_BITS, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=1e-6
    )

    n_tr = len(train_idx)
    n_batches = math.ceil(n_tr / cfg.batch_size)
    best_val_mae = float("inf")
    best_state = None

    print(f"\nTraining for {cfg.epochs} epochs, batch_size={cfg.batch_size} ...")
    print("-" * 60)

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        # Shuffle training data.
        perm = torch.randperm(n_tr)
        X_tr_sh = X_tr[perm].to(device)
        y_tr_sh = y_tr[perm].to(device)

        train_loss = 0.0
        for b in range(n_batches):
            s = b * cfg.batch_size
            e = min(s + cfg.batch_size, n_tr)
            xb = X_tr_sh[s:e]
            yb = y_tr_sh[s:e]

            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * (e - s)

        train_loss /= n_tr
        scheduler.step()

        # Validation.
        model.eval()
        with torch.no_grad():
            pred_va = model(X_va.to(device)).cpu().numpy().ravel()

        y_va_raw = y_va.numpy().ravel() * PIC50_STD + PIC50_MEAN
        pred_va_raw = pred_va * PIC50_STD + PIC50_MEAN
        val_mae = float(mean_absolute_error(y_va_raw, pred_va_raw))
        val_r2 = float(r2_score(y_va_raw, pred_va_raw))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:4d}/{cfg.epochs} | "
                f"train MSE={train_loss:.4f} | "
                f"val MAE={val_mae:.4f} | "
                f"val R²={val_r2:.4f}"
            )

    # --------------------------------------------------------
    # Evaluate best checkpoint on untouched scaffold test set
    # --------------------------------------------------------
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_te = model(X_te.to(device)).cpu().numpy().ravel()

    y_te_raw = y_te.numpy().ravel() * PIC50_STD + PIC50_MEAN
    pred_te_raw = pred_te * PIC50_STD + PIC50_MEAN

    test_mae = float(mean_absolute_error(y_te_raw, pred_te_raw))
    test_r2 = float(r2_score(y_te_raw, pred_te_raw))

    print()
    print("=" * 60)
    print("SCAFFOLD TEST RESULTS")
    print("=" * 60)
    print(f"  Test R²  : {test_r2:.4f}  (ExtraTrees baseline: 0.6156)")
    print(f"  Test MAE : {test_mae:.4f}  (ExtraTrees baseline: 0.5879)")

    if test_r2 < 0.40:
        print(
            "\nWARNING: Test R² < 0.40. The surrogate may not provide a "
            "reliable gradient signal. Consider increasing --epochs or "
            "--hidden_size before using this surrogate in train_joint_surrogate.py."
        )
    else:
        print("\nSurrogate R² is acceptable. Ready to use as a generator loss term.")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    model_path = out / "bace1_surrogate.pt"
    torch.save(best_state, model_path)

    metadata = {
        "architecture": "MLP 2048->512->256->128->1",
        "n_bits": N_BITS,
        "radius": RADIUS,
        "dropout": cfg.dropout,
        "output": "normalized_pIC50",
        "pIC50_mean": PIC50_MEAN,
        "pIC50_std": PIC50_STD,
        "unique_molecules": len(df),
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "extratrees_baseline_r2": 0.6156,
        "extratrees_baseline_mae": 0.5879,
    }

    (out / "surrogate_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print(f"\nSaved surrogate model  : {model_path}")
    print(f"Saved surrogate metadata: {out / 'surrogate_metadata.json'}")

    return test_r2


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train differentiable MLP surrogate of BACE1 ExtraTrees predictor."
    )
    parser.add_argument("--input", required=True,
                        help="Path to bace1_continuous_bbb.csv")
    parser.add_argument("--output_dir", default="results/bace1_surrogate_model")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs (default 200; surrogate is fast to train).")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()
    train(args)
