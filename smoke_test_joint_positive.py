"""
Smoke test for train_joint_positive_disc.

Verifies:
  1. The training script can be imported without errors.
  2. A 10-step training loop completes with batch_size=8.
  3. G_loss and D_loss are finite after 10 steps.
  4. At least 1 valid molecule (>3 heavy atoms) is produced.
  5. No NaN gradients in generator parameters.

Results are written to logs/joint_positive_smoke.txt.
Exits with code 0 on PASS, code 1 on FAIL.
"""

import math
import os
import sys

# ── ensure project root is on the import path ──────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── create logs directory before anything else ─────────────────────────────
LOGS_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOGS_DIR, "joint_positive_smoke.txt")


def write_log(lines: list[str]) -> None:
    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ── Step 1: import check ────────────────────────────────────────────────────
try:
    import train_joint_positive_disc  # noqa: F401
except Exception as exc:
    msg = f"Import failed: {exc}"
    print(f"SMOKE TEST FAILED: {msg}")
    write_log(
        [
            "SMOKE TEST: FAILED",
            "G_loss_final: N/A",
            "D_loss_final: N/A",
            "valid_molecules_in_sample: N/A",
            "nan_gradients: N/A",
            f"detail: {msg}",
        ]
    )
    sys.exit(1)

print("Import check: OK")

# ── Remaining imports (only after import-check passes) ─────────────────────
import random
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.model.conditional_models import (
    ConditionalGenerator,
    ConditionalDiscriminator,
    relax_graph,
)

# ── Configuration ───────────────────────────────────────────────────────────
BATCH_SIZE = 8
MAX_ATOM = 45
CONDITION_DIM = 3
SEED = 1
N_STEPS = 10

CHEMBL_SMI = os.path.join(ROOT, "data", "chembl_train.smi")
JP_SMI = os.path.join(ROOT, "druggen_data", "bace1_joint_positive_train.smi")
# Use the full BACE1 SMI for vocabulary building — this matches the
# baseline checkpoints (Issue3-ChemistryReadout) which were trained with
# bace1.smi + chembl_train.smi (node_dim=14).  The joint-positive subset
# is a strict subset of BACE1 and does not cover all atom types, so
# building a vocab from it alone yields node_dim=11, which mismatches the
# warm-start checkpoint tensors.
BACE1_SMI = os.path.join(ROOT, "druggen_data", "bace1.smi")
INIT_GEN = os.path.join(
    ROOT, "experiments", "models", "Issue3-ChemistryReadout", "joint_generator.pt"
)
INIT_DISC = os.path.join(
    ROOT, "experiments", "models", "Issue3-ChemistryReadout", "joint_discriminator.pt"
)

# ── Reproducibility ─────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Encoders / decoders ──────────────────────────────────────────────────────
print("Loading encoders/decoders ...")
# Use the full bace1.smi (not the joint-positive subset) to match the
# vocabulary of the warm-start checkpoints (node_dim=14, edge_dim=5).
atom_enc, atom_dec, bond_enc, bond_dec = get_encoders_decoders(
    CHEMBL_SMI, BACE1_SMI, MAX_ATOM
)

# ── Processed .pt filenames ─────────────────────────────────────────────────
# The ChEMBL dataset uses the cached chembl_train45.pt (already on disk).
# The joint-positive dataset is filtered using the full BACE1 vocabulary,
# so the cached .pt file name encodes both file stems.
chembl_pt_name = os.path.basename(CHEMBL_SMI).replace(".smi", "") + str(MAX_ATOM) + ".pt"
# jp_dataset uses the JP SMI file but is processed with the bace1_chembl vocab
# The pt file is stored in druggen_data/ using only the JP stem for uniqueness.
jp_pt_name = os.path.basename(JP_SMI).replace(".smi", "") + str(MAX_ATOM) + ".pt"

# ── Datasets ─────────────────────────────────────────────────────────────────
print("Building/loading ChEMBL dataset ...")
general = DruggenDataset(
    "data",
    chembl_pt_name,
    CHEMBL_SMI,
    MAX_ATOM,
    False,
    atom_enc,
    atom_dec,
    bond_enc,
    bond_dec,
)

print("Building/loading joint-positive dataset ...")
jp_dataset = DruggenDataset(
    "druggen_data",
    jp_pt_name,
    JP_SMI,
    MAX_ATOM,
    False,
    atom_enc,
    atom_dec,
    bond_enc,
    bond_dec,
)

print(f"General graphs : {len(general):,}")
print(f"Joint-positive : {len(jp_dataset):,}")

# ── DataLoaders ───────────────────────────────────────────────────────────────
loader = DataLoader(
    general, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
target_loader = DataLoader(
    jp_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)

# ── Graph dimensions ──────────────────────────────────────────────────────────
m_dim = len(atom_dec)
b_dim = len(bond_dec)
vertexes = int(general[0].x.shape[0])
print(f"m_dim={m_dim}  b_dim={b_dim}  vertexes={vertexes}")

# ── Models ────────────────────────────────────────────────────────────────────
G = ConditionalGenerator(
    "relu", vertexes, b_dim, m_dim, 0.0,
    dim=128, depth=1, heads=8, mlp_ratio=3,
    condition_dim=CONDITION_DIM,
).to(device)

D = ConditionalDiscriminator(
    "relu", vertexes, b_dim, m_dim, 0.0,
    dim=128, depth=1, heads=8, mlp_ratio=3,
    condition_dim=CONDITION_DIM,
).to(device)

# ── Warm-start checkpoints ────────────────────────────────────────────────────
train_joint_positive_disc.load_model_checkpoint(G, INIT_GEN, device, "generator")
train_joint_positive_disc.load_model_checkpoint(D, INIT_DISC, device, "discriminator")


def critic(edge, node, cond):
    return D(edge, node, cond, return_property=False)[0]


# ── Optimizers ────────────────────────────────────────────────────────────────
go = torch.optim.AdamW(G.parameters(), lr=1e-5, betas=(0.9, 0.999))
do = torch.optim.AdamW(D.parameters(), lr=1e-5, betas=(0.9, 0.999))

# ── Training loop (10 steps) ──────────────────────────────────────────────────
G.train()
D.train()

target_iter = iter(target_loader)
main_iter = iter(loader)

g_loss_final = float("nan")
d_loss_final = float("nan")

print("Running 10 training steps ...")
for step in range(N_STEPS):

    # ── fetch batches ──────────────────────────────────────────────────────
    try:
        mol_batch = next(main_iter)
    except StopIteration:
        main_iter = iter(loader)
        mol_batch = next(main_iter)

    try:
        real_batch = next(target_iter)
    except StopIteration:
        target_iter = iter(target_loader)
        real_batch = next(target_iter)

    # ── load graph tensors ────────────────────────────────────────────────
    _, mol_adj, mol_x = load_molecules(
        mol_batch, b_dim=b_dim, m_dim=m_dim,
        device=device, batch_size=BATCH_SIZE,
    )
    _, real_adj, real_x = load_molecules(
        real_batch, b_dim=b_dim, m_dim=m_dim,
        device=device, batch_size=BATCH_SIZE,
    )

    zero_cond = torch.zeros(BATCH_SIZE, CONDITION_DIM, device=device)

    # ── Discriminator step ────────────────────────────────────────────────
    D.zero_grad(set_to_none=True)

    with torch.no_grad():
        _, _, node_d, edge_d = G(mol_adj, mol_x, zero_cond)

    fake_edge_d, fake_node_d = relax_graph(node_d, edge_d, mode="st")

    real_score = critic(real_adj, real_x, zero_cond)
    fake_score = critic(fake_edge_d, fake_node_d, zero_cond)

    eps_n = torch.rand(BATCH_SIZE, 1, 1, device=device)
    eps_e = torch.rand(BATCH_SIZE, 1, 1, 1, device=device)

    interp_node = (eps_n * real_x + (1.0 - eps_n) * fake_node_d).requires_grad_(True)
    interp_edge = (eps_e * real_adj + (1.0 - eps_e) * fake_edge_d).requires_grad_(True)

    interp_score = critic(interp_edge, interp_node, zero_cond)

    grads = torch.autograd.grad(
        interp_score,
        [interp_node, interp_edge],
        torch.ones_like(interp_score),
        create_graph=True,
    )
    gradient = torch.cat([grads[0].flatten(1), grads[1].flatten(1)], dim=1)
    gp = ((gradient.norm(2, dim=1) - 1.0) ** 2).mean()

    d_loss = fake_score.mean() - real_score.mean() + 10.0 * gp
    d_loss.backward()
    do.step()

    # ── Generator step ────────────────────────────────────────────────────
    G.zero_grad(set_to_none=True)

    _, _, node_g, edge_g = G(mol_adj, mol_x, zero_cond)
    fake_edge_g, fake_node_g = relax_graph(node_g, edge_g, mode="st")

    g_loss = -critic(fake_edge_g, fake_node_g, zero_cond).mean()
    g_loss.backward()
    go.step()

    g_loss_final = float(g_loss.detach())
    d_loss_final = float(d_loss.detach())
    print(f"  step {step+1:2d}/10  G={g_loss_final:.4f}  D={d_loss_final:.4f}")

# ── Check 3a: finite losses ──────────────────────────────────────────────────
g_finite = math.isfinite(g_loss_final)
d_finite = math.isfinite(d_loss_final)

# ── Check 3c: at least 1 valid molecule ──────────────────────────────────────
print("Evaluating molecule validity ...")
G.eval()
with torch.no_grad():
    _, _, ns_eval, es_eval = G(mol_adj, mol_x, zero_cond)
G.train()

valid_count = 0
sample_size = BATCH_SIZE
for i in range(sample_size):
    try:
        node_np = ns_eval[i].argmax(dim=-1).cpu().numpy()
        edge_np = es_eval[i].argmax(dim=-1).cpu().numpy()
        mol = jp_dataset.matrices2mol(node_np, edge_np, strict=True)
        if mol is not None and mol.GetNumHeavyAtoms() > 3:
            valid_count += 1
    except Exception:
        pass

# ── Check 3d: NaN gradients in generator ────────────────────────────────────
# Run one more forward/backward to get gradients on the generator
G.zero_grad(set_to_none=True)

try:
    mol_batch2 = next(main_iter)
except StopIteration:
    main_iter = iter(loader)
    mol_batch2 = next(main_iter)

_, mol_adj2, mol_x2 = load_molecules(
    mol_batch2, b_dim=b_dim, m_dim=m_dim,
    device=device, batch_size=BATCH_SIZE,
)
zero_cond2 = torch.zeros(BATCH_SIZE, CONDITION_DIM, device=device)
_, _, node_g2, edge_g2 = G(mol_adj2, mol_x2, zero_cond2)
fake_edge_g2, fake_node_g2 = relax_graph(node_g2, edge_g2, mode="st")
g_loss_grad = -critic(fake_edge_g2, fake_node_g2, zero_cond2).mean()
g_loss_grad.backward()

nan_gradients = False
for name, param in G.named_parameters():
    if param.grad is not None and param.grad.isnan().any():
        nan_gradients = True
        print(f"  NaN gradient in: {name}")
        break

# ── Evaluate pass/fail ────────────────────────────────────────────────────────
failures = []
if not g_finite:
    failures.append(f"G_loss is not finite: {g_loss_final}")
if not d_finite:
    failures.append(f"D_loss is not finite: {d_loss_final}")
if valid_count < 1:
    failures.append(f"No valid molecules in decoded sample (got {valid_count})")
if nan_gradients:
    failures.append("NaN gradients detected in generator parameters")

passed = len(failures) == 0
status = "PASSED" if passed else "FAILED"
detail = "; ".join(failures) if failures else "all checks passed"

# ── Write log ────────────────────────────────────────────────────────────────
log_lines = [
    f"SMOKE TEST: {status}",
    f"G_loss_final: {g_loss_final}",
    f"D_loss_final: {d_loss_final}",
    f"valid_molecules_in_sample: {valid_count}",
    f"nan_gradients: {nan_gradients}",
    f"detail: {detail}",
]
write_log(log_lines)

for line in log_lines:
    print(line)

if passed:
    print("SMOKE TEST PASSED. Proceeding to pilot.")
    sys.exit(0)
else:
    print(f"SMOKE TEST FAILED: {detail}")
    sys.exit(1)
