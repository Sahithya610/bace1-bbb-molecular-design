"""
Post-hoc analysis of the smoke-test generation sample.
Reproduces the exact same seed / checkpoint / batch that the smoke test used,
then reports:
  - atom-class histogram (including padding class 0)
  - per-molecule validity
  - SMILES of any valid molecules
  - Tanimoto max/mean to the joint-positive training set
"""

import os, sys, random
import numpy as np
import torch
from collections import Counter
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.model.conditional_models import ConditionalGenerator, ConditionalDiscriminator, relax_graph
import train_joint_positive_disc

# ── exact same config as smoke test ─────────────────────────────────────────
BATCH_SIZE  = 8
MAX_ATOM    = 45
CONDITION_DIM = 3
SEED        = 1
N_STEPS     = 10   # re-run the same 10 steps to reach the same model state

CHEMBL_SMI  = os.path.join(ROOT, "data",        "chembl_train.smi")
JP_SMI      = os.path.join(ROOT, "druggen_data","bace1_joint_positive_train.smi")
BACE1_SMI   = os.path.join(ROOT, "druggen_data","bace1.smi")
INIT_GEN    = os.path.join(ROOT, "experiments","models","Issue3-ChemistryReadout","joint_generator.pt")
INIT_DISC   = os.path.join(ROOT, "experiments","models","Issue3-ChemistryReadout","joint_discriminator.pt")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

atom_enc, atom_dec, bond_enc, bond_dec = get_encoders_decoders(CHEMBL_SMI, BACE1_SMI, MAX_ATOM)

chembl_pt = os.path.basename(CHEMBL_SMI).replace(".smi","") + str(MAX_ATOM) + ".pt"
jp_pt     = os.path.basename(JP_SMI   ).replace(".smi","") + str(MAX_ATOM) + ".pt"

general = DruggenDataset("data",       chembl_pt, CHEMBL_SMI, MAX_ATOM, False, atom_enc, atom_dec, bond_enc, bond_dec)
jp_ds   = DruggenDataset("druggen_data", jp_pt,   JP_SMI,    MAX_ATOM, False, atom_enc, atom_dec, bond_enc, bond_dec)

loader        = DataLoader(general, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
target_loader = DataLoader(jp_ds,   batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

m_dim = len(atom_dec); b_dim = len(bond_dec); vertexes = int(general[0].x.shape[0])
print(f"m_dim={m_dim}  b_dim={b_dim}  vertexes={vertexes}")

G = ConditionalGenerator( "relu", vertexes, b_dim, m_dim, 0.0, dim=128, depth=1, heads=8, mlp_ratio=3, condition_dim=CONDITION_DIM).to(device)
D = ConditionalDiscriminator("relu", vertexes, b_dim, m_dim, 0.0, dim=128, depth=1, heads=8, mlp_ratio=3, condition_dim=CONDITION_DIM).to(device)
train_joint_positive_disc.load_model_checkpoint(G, INIT_GEN,  device, "generator")
train_joint_positive_disc.load_model_checkpoint(D, INIT_DISC, device, "discriminator")

def critic(edge, node, cond): return D(edge, node, cond, return_property=False)[0]

go = torch.optim.AdamW(G.parameters(), lr=1e-5, betas=(0.9,0.999))
do = torch.optim.AdamW(D.parameters(), lr=1e-5, betas=(0.9,0.999))
G.train(); D.train()

target_iter = iter(target_loader); main_iter = iter(loader)
mol_adj = mol_x = None

for step in range(N_STEPS):
    try:   mol_batch  = next(main_iter)
    except StopIteration: main_iter = iter(loader); mol_batch = next(main_iter)
    try:   real_batch = next(target_iter)
    except StopIteration: target_iter = iter(target_loader); real_batch = next(target_iter)

    _, mol_adj, mol_x = load_molecules(mol_batch,  b_dim=b_dim, m_dim=m_dim, device=device, batch_size=BATCH_SIZE)
    _, real_adj, real_x = load_molecules(real_batch, b_dim=b_dim, m_dim=m_dim, device=device, batch_size=BATCH_SIZE)
    zero_cond = torch.zeros(BATCH_SIZE, CONDITION_DIM, device=device)

    D.zero_grad(set_to_none=True)
    with torch.no_grad(): _, _, nd, ed = G(mol_adj, mol_x, zero_cond)
    fe_d, fn_d = relax_graph(nd, ed, mode="st")
    rs = critic(real_adj, real_x, zero_cond); fs = critic(fe_d, fn_d, zero_cond)
    eps_n = torch.rand(BATCH_SIZE,1,1,device=device); eps_e = torch.rand(BATCH_SIZE,1,1,1,device=device)
    in_ = (eps_n*real_x+(1-eps_n)*fn_d).requires_grad_(True); ie_ = (eps_e*real_adj+(1-eps_e)*fe_d).requires_grad_(True)
    grads = torch.autograd.grad(critic(ie_,in_,zero_cond),[in_,ie_],torch.ones(BATCH_SIZE,device=device),create_graph=True)
    gp = ((torch.cat([grads[0].flatten(1),grads[1].flatten(1)],1).norm(2,1)-1)**2).mean()
    (fs.mean()-rs.mean()+10.*gp).backward(); do.step()

    G.zero_grad(set_to_none=True)
    _, _, ng, eg = G(mol_adj, mol_x, zero_cond); feg, fng = relax_graph(ng, eg, mode="st")
    (-critic(feg, fng, zero_cond).mean()).backward(); go.step()

# ── eval pass: same batch as smoke test ─────────────────────────────────────
G.eval()
with torch.no_grad():
    _, _, ns, es = G(mol_adj, mol_x, zero_cond)
G.train()

# ── atom-class distribution ──────────────────────────────────────────────────
node_classes = ns.argmax(dim=-1).cpu().numpy()   # [8, 45]
flat_classes  = node_classes.flatten()            # 360 positions
class_counts  = Counter(flat_classes.tolist())

print("\n── Atom-class distribution (all 8×45 = 360 node positions) ──")
print(f"  class 0 (PAD): {class_counts.get(0,0):4d}  ({class_counts.get(0,0)/360*100:.1f}%)")
non_pad = {c:v for c,v in class_counts.items() if c!=0}
for cls in sorted(non_pad):
    atomic_num = atom_dec.get(cls, "?")
    el = Chem.GetPeriodicTable().GetElementSymbol(atomic_num) if isinstance(atomic_num,int) and atomic_num>0 else "?"
    print(f"  class {cls:2d} (atomic {atomic_num:3}, {el}): {non_pad[cls]:4d}  ({non_pad[cls]/360*100:.1f}%)")

# ── per-molecule decode ───────────────────────────────────────────────────────
print(f"\n── Per-molecule decode (sample_size = {BATCH_SIZE}) ──")
valid_smiles = []
for i in range(BATCH_SIZE):
    np_node = ns[i].argmax(dim=-1).cpu().numpy()
    np_edge = es[i].argmax(dim=-1).cpu().numpy()
    non_pad_count = int((np_node != 0).sum())
    mol = jp_ds.matrices2mol(np_node, np_edge, strict=True)
    if mol is None:
        status = "INVALID (sanitization failed)"
        heavy = 0
        smi   = "N/A"
    else:
        heavy = mol.GetNumHeavyAtoms()
        smi   = Chem.MolToSmiles(mol, canonical=True)
        status = f"VALID  heavy={heavy}" if heavy > 3 else f"TRIVIAL heavy={heavy}"
        if heavy > 3:
            valid_smiles.append(smi)
    print(f"  mol[{i}]: non-pad nodes={non_pad_count:2d}  {status}  {smi[:60] if smi!='N/A' else ''}")

validity_frac = len(valid_smiles) / BATCH_SIZE
print(f"\nValidity fraction: {len(valid_smiles)}/{BATCH_SIZE} = {validity_frac:.3f}")

# ── Tanimoto check ────────────────────────────────────────────────────────────
train_fps = []
with open(JP_SMI, encoding="utf-8") as fh:
    for line in fh:
        s = line.strip()
        if not s: continue
        m = Chem.MolFromSmiles(s)
        if m:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
            train_fps.append((s, fp))

print(f"\n── Tanimoto check vs {len(train_fps)} JP training molecules ──")
for smi in valid_smiles:
    mol_q = Chem.MolFromSmiles(smi)
    if mol_q is None: continue
    fp_q = AllChem.GetMorganFingerprintAsBitVect(mol_q, 2, 2048)
    sims = DataStructs.BulkTanimotoSimilarity(fp_q, [fp for _,fp in train_fps])
    best_idx = int(np.argmax(sims)); best_tc = sims[best_idx]
    exact = (sims[best_idx] == 1.0)
    print(f"  generated : {smi[:70]}")
    print(f"  best match: {train_fps[best_idx][0][:70]}")
    print(f"  Tc={best_tc:.4f}  exact_duplicate={exact}")

print("\nAnalysis complete.")
