r"""
diagnose_generation_graphs.py  (READ-ONLY; Steps 3 and 5 of the handoff)

Runs the REAL ConditionalGenerator forward pass (no re-implemented layers) on
the same ChEMBL input graphs under several requested pIC50 values, then uses
the same decode pipeline as generate_joint_molecules.py, with bookkeeping.

    python diagnose_generation_graphs.py --num_batches 2          # quick smoke test
    python diagnose_generation_graphs.py                          # 1024 graphs, pIC50 5.5 vs 8.0
    python diagnose_generation_graphs.py --inputs random --seed 1 # random ChEMBL inputs

It reports, per requested condition:
  * how many raw graphs sanitize, decode cleanly, or are only "rescued" by the
    fragment-dropping and '*' -> 'C' steps of the generation script
  * padding bookkeeping: padding nodes bonded to real atoms, graphs with no real atoms
  * argmax node-class shares BEFORE any rescue, and element presence in final molecules
  * similarity of each generated graph to its own ChEMBL input graph (copy check)
and, for the first two conditions, paired logit / decision differences.

Generation in generate_joint_molecules.py is deterministic (eval mode, dropout 0,
argmax, shuffle=False), so no RNG control is needed; --seed only affects
--inputs random. Writes only new files: results\diag_<tag>_graphs.csv and
results\diag_<tag>_summary.txt. Checkpoints and mappings are never modified.
"""
import argparse
import collections
import os

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Pure numpy helpers (no torch / RDKit needed)
# ----------------------------------------------------------------------

def tril_pairs(v):
    return np.tril_indices(v, k=-1)  # row > col, same convention as matrices2mol


def top2_margin(logits):
    s = np.sort(logits, axis=-1)
    return s[..., -1] - s[..., -2]


def graph_counts(n, e):
    """n: [V] atom class labels, e: [V, V] bond class labels (class 0 = pad / no-bond)."""
    il, jl = tril_pairs(len(n))
    bonded = e[il, jl] != 0
    pad_i, pad_j = n[il] == 0, n[jl] == 0
    return dict(
        real_atoms=int((n != 0).sum()),
        pad_nodes=int((n == 0).sum()),
        real_real_bonds=int((bonded & ~pad_i & ~pad_j).sum()),
        pad_real_bonds=int((bonded & (pad_i ^ pad_j)).sum()),
        pad_pad_bonds=int((bonded & pad_i & pad_j).sum()),
    )


def input_similarity(out_n, in_n, out_e, in_e):
    """Per-graph agreement between generated and input graphs.
    out_n/in_n: [B, V]; out_e/in_e: [B, V, V] (class labels)."""
    il, jl = tril_pairs(out_n.shape[1])
    node_agree = (out_n == in_n).mean(axis=1)
    oe, ie = out_e[:, il, jl], in_e[:, il, jl]
    inter = ((oe == ie) & (oe != 0)).sum(axis=1)
    union = ((oe != 0) | (ie != 0)).sum(axis=1)
    jaccard = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return node_agree, jaccard


class PairedStats:
    """Accumulates logit and decision differences between condition A and B."""

    def __init__(self):
        self.t = collections.Counter()
        self.max_abs = {"node": 0.0, "edge": 0.0}
        self.graphs = 0

    def add(self, nA, nB, eA, eB):
        il, jl = tril_pairs(nA.shape[1])
        self.graphs += nA.shape[0]
        for name, a, b in (("node", nA, nB), ("edge", eA[:, il, jl, :], eB[:, il, jl, :])):
            flips = a.argmax(-1) != b.argmax(-1)
            d = np.abs(a - b)
            m = top2_margin(a)
            self.t[name + "_flips"] += int(flips.sum())
            self.t[name + "_positions"] += int(flips.size)
            self.t[name + "_abs_sum"] += float(d.sum())
            self.t[name + "_abs_n"] += int(d.size)
            self.t[name + "_margin_sum"] += float(m.sum())
            self.t[name + "_margin_n"] += int(m.size)
            self.t[name + "_margin_lt_0.25"] += int((m < 0.25).sum())
            self.t[name + "_graphs_with_flip"] += int(flips.any(axis=tuple(range(1, flips.ndim))).sum())
            self.max_abs[name] = max(self.max_abs[name], float(d.max()))

    def lines(self, label_a, label_b):
        t = self.t
        out = [f"Paired logit comparison, same input graphs: pIC50 {label_a} (A) vs {label_b} (B); "
               f"{self.graphs} graphs"]
        for name, what in (("node", "atom decisions"), ("edge", "bond decisions (lower triangle)")):
            pos = max(t[name + "_positions"], 1)
            out.append(f"  {what}: flipped {t[name + '_flips']}/{pos} ({100 * t[name + '_flips'] / pos:.4f}%); "
                       f"graphs with >=1 flip: {t[name + '_graphs_with_flip']}/{self.graphs}")
            out.append(f"    mean |logit change| per logit: {t[name + '_abs_sum'] / max(t[name + '_abs_n'], 1):.5f}; "
                       f"max: {self.max_abs[name]:.5f}")
            out.append(f"    mean top1-top2 margin under A: {t[name + '_margin_sum'] / pos:.4f}; "
                       f"positions with margin < 0.25: {100 * t[name + '_margin_lt_0.25'] / pos:.3f}%")
        return out


def summarize(df, class_names, label):
    """df: per-graph records for ONE requested condition."""
    n = len(df)
    valid = df["final_smiles"].notna()
    clean = df["sanitize_ok"] & (df["n_real_frag"] == 1) & (df["dummies_in_real"] == 0)
    L = [f"--- requested pIC50 = {label} ---",
         f"graphs: {n}",
         f"raw graph sanitizes (all 45 positions, before any rescue): {int(df['sanitize_ok'].sum())} ({100 * df['sanitize_ok'].mean():.2f}%)",
         f"clean decode (sanitizes, 1 real fragment, no padding bonded in): {int(clean.sum())} ({100 * clean.mean():.2f}%)",
         f"final valid after generation-script pipeline: {int(valid.sum())} ({100 * valid.mean():.2f}%)",
         f"  unique among final valid: {df.loc[valid, 'final_smiles'].nunique()}",
         f"  valid only thanks to dropping extra real fragments: {int((valid & (df['n_real_frag'] > 1)).sum())}",
         f"  valid only thanks to '*' -> 'C' (padding bonded to real atoms): {int((valid & (df['dummies_in_real'] > 0)).sum())}",
         f"graphs with padding nodes bonded to real atoms: {int((df['pad_real_bonds'] > 0).sum())} ({100 * (df['pad_real_bonds'] > 0).mean():.2f}%)",
         f"graphs with no real atoms at all (pipeline returns 'C'): {int((df['real_atoms'] == 0).sum())}",
         f"mean real atoms {df['real_atoms'].mean():.2f} | padding nodes {df['pad_nodes'].mean():.2f} | "
         f"real-real bonds {df['real_real_bonds'].mean():.2f} | pad-real bonds {df['pad_real_bonds'].mean():.3f}"]
    cls_cols = [c for c in df.columns if c.startswith("raw_class_")]
    tot = sum(df[c].sum() for c in cls_cols if not c.endswith("_pad"))
    shares = ", ".join(f"{c.split('_', 3)[3]} {100 * df[c].sum() / max(tot, 1):.2f}%"
                       for c in cls_cols if not c.endswith("_pad") and df[c].sum() > 0)
    L.append(f"argmax atom-class share among real nodes (before any rescue): {shares}")
    raw_o = [c for c in cls_cols if c.endswith("_O")]
    if raw_o:
        L.append(f"raw graphs containing >=1 oxygen node: {int((df[raw_o[0]] > 0).sum())} ({100 * (df[raw_o[0]] > 0).mean():.2f}%)")
    if valid.any():
        v = df[valid]
        L.append("final valid molecules containing: " +
                 ", ".join(f"{e} {100 * v['has_' + e].mean():.2f}%" for e in ("C", "N", "O", "S", "F", "Cl")) +
                 f" | mean heavy atoms {v['final_heavy_atoms'].mean():.2f}")
    L.append(f"similarity to own ChEMBL input: mean node-label agreement {df['node_agree_input'].mean():.4f}, "
             f"mean bond Jaccard {df['bond_jaccard_input'].mean():.4f}, "
             f"exact SMILES copies of the input {int(df['exact_copy_of_input'].sum())} ({100 * df['exact_copy_of_input'].mean():.2f}%)")
    return L


# ----------------------------------------------------------------------
# Main (needs the project environment)
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="experiments/models/Issue3-ChemistryReadout/joint_generator.pt")
    ap.add_argument("--pic50", type=float, nargs="+", default=[5.5, 8.0],
                    help="requested pIC50 values; the first two are compared pairwise")
    ap.add_argument("--logbb", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_batches", type=int, default=32)
    ap.add_argument("--inputs", choices=["first", "random"], default="first",
                    help="'first' = same first ChEMBL graphs as generate_joint_molecules.py")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default=None, help="output file tag (default: checkpoint folder name)")
    args = ap.parse_args()

    import torch
    from rdkit import Chem, RDLogger
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    from generate_joint_molecules import MAX_ATOMS, build_model, canonical_smiles, make_condition
    from src.data.dataset import DruggenDataset
    from src.data.utils import get_encoders_decoders, load_molecules

    RDLogger.DisableLog("rdApp.*")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or os.path.basename(os.path.dirname(args.checkpoint)) or "run"

    atom_encoder, atom_decoder, bond_encoder, bond_decoder = get_encoders_decoders(
        "data/chembl_train.smi", "druggen_data/bace1.smi", MAX_ATOMS)
    dataset = DruggenDataset(
        root="data", dataset_file="chembl_train45.pt",
        raw_files=["chembl_train.smi", "pre_filter.pt", "pre_transform.pt"],
        max_atom=MAX_ATOMS, features=False,
        atom_encoder=atom_encoder, atom_decoder=atom_decoder,
        bond_encoder=bond_encoder, bond_decoder=bond_decoder)
    vertexes = int(dataset[0].x.shape[0])
    nodes = int(dataset[0].x.shape[1])
    edges = len(bond_decoder)
    print(f"Device {device} | ChEMBL graphs {len(dataset)} | vertexes/nodes/edges {vertexes}/{nodes}/{edges}")

    pt = Chem.GetPeriodicTable()
    names = {k: ("pad" if z == 0 else pt.GetElementSymbol(int(z))) for k, z in atom_decoder.items()}

    model = build_model(device=device, vertexes=vertexes, edges=edges, nodes=nodes, checkpoint=args.checkpoint)
    print("Checkpoint loaded:", args.checkpoint)

    n_graphs = args.batch_size * args.num_batches
    if args.inputs == "random":
        idx = np.random.default_rng(args.seed).choice(len(dataset), size=min(n_graphs, len(dataset)), replace=False)
        source = Subset(dataset, idx.tolist())
    else:
        source = dataset
    loader = DataLoader(source, batch_size=args.batch_size, shuffle=False, drop_last=False)

    def labels(t, ndim):
        a = t.detach().cpu().numpy()
        return a.argmax(-1) if a.ndim == ndim else a.astype(np.int64)

    def decode(n_lab, e_lab):
        """Same steps as generate_joint_molecules.py, plus structural bookkeeping."""
        mol = dataset.matrices2mol(n_lab, e_lab, strict=False)
        frags = Chem.GetMolFrags(mol)
        real = [f for f in frags if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 0 for i in f)]
        dummies = sum(1 for f in real for i in f if mol.GetAtomWithIdx(i).GetAtomicNum() == 0)
        s = canonical_smiles(mol)  # sanitizes in place
        sanitize_ok = s is not None
        if s is not None and "." in s:
            s = max(s.split("."), key=len)
            s = canonical_smiles(Chem.MolFromSmiles(s))
        if s is not None and "*" in s:
            s = s.replace("*", "C")
            s = canonical_smiles(Chem.MolFromSmiles(s))
        return s, dict(sanitize_ok=sanitize_ok, n_real_frag=len(real), dummies_in_real=dummies)

    conds = list(args.pic50)
    records = {p: [] for p in conds}
    paired = PairedStats()
    seen = 0

    with torch.inference_mode():
        for b, mol_batch in enumerate(loader):
            if b >= args.num_batches:
                break
            B = mol_batch.num_graphs
            _, a_tensor, x_tensor = load_molecules(data=mol_batch, batch_size=B, device=device,
                                                   b_dim=edges, m_dim=nodes)
            in_n, in_e = labels(x_tensor, 3), labels(a_tensor, 4)
            in_smiles = [decode(in_n[k], in_e[k])[0] for k in range(B)]
            kept = {}
            for p in conds:
                cond = make_condition(B, p, args.logbb, device)
                _, _, nl, el = model(a_tensor, x_tensor, cond)
                nl, el = nl.float().cpu().numpy(), el.float().cpu().numpy()
                if len(kept) < 2:
                    kept[p] = (nl, el)
                out_n, out_e = nl.argmax(-1), el.argmax(-1)
                node_agree, jac = input_similarity(out_n, in_n, out_e, in_e)
                for k in range(B):
                    s, info = decode(out_n[k], out_e[k])
                    rec = dict(requested_pIC50=p, index=seen + k, input_smiles=in_smiles[k], final_smiles=s)
                    rec.update(info)
                    rec.update(graph_counts(out_n[k], out_e[k]))
                    counts = np.bincount(out_n[k], minlength=nodes)
                    for c in range(nodes):
                        rec[f"raw_class_{c}_{names[c]}"] = int(counts[c])
                    m = Chem.MolFromSmiles(s) if s else None
                    elems = {a.GetSymbol() for a in m.GetAtoms()} if m is not None else set()
                    for e_sym in ("C", "N", "O", "S", "F", "Cl"):
                        rec["has_" + e_sym] = e_sym in elems
                    rec["final_heavy_atoms"] = m.GetNumHeavyAtoms() if m is not None else 0
                    rec["node_agree_input"] = float(node_agree[k])
                    rec["bond_jaccard_input"] = float(jac[k])
                    rec["exact_copy_of_input"] = bool(s is not None and s == in_smiles[k])
                    records[p].append(rec)
            if len(conds) >= 2:
                (nA, eA), (nB, eB) = kept[conds[0]], kept[conds[1]]
                paired.add(nA, nB, eA, eB)
            seen += B
            print(f"batch {b + 1}/{args.num_batches} | graphs {seen}")

    all_df = pd.concat([pd.DataFrame(r) for r in records.values()], ignore_index=True)
    lines = [f"Checkpoint: {args.checkpoint}", f"Inputs: {args.inputs} ({seen} ChEMBL graphs) | logBB {args.logbb}", ""]
    for p in conds:
        lines += summarize(all_df[all_df["requested_pIC50"] == p], names, p) + [""]
    if len(conds) >= 2:
        lines += paired.lines(conds[0], conds[1])
        a = all_df[all_df["requested_pIC50"] == conds[0]].set_index("index")["final_smiles"]
        bb = all_df[all_df["requested_pIC50"] == conds[1]].set_index("index")["final_smiles"]
        va, vb = a.notna(), bb.notna()
        lines += [f"  final SMILES identical in both conditions: {int(((a == bb) & va & vb).sum())}/{len(a)}; "
                  f"differ (both valid): {int(((a != bb) & va & vb).sum())}; valid in only one: {int((va ^ vb).sum())}"]

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"diag_{tag}_graphs.csv")
    txt_path = os.path.join("results", f"diag_{tag}_summary.txt")
    all_df.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nSaved: {csv_path}\nSaved: {txt_path}")


if __name__ == "__main__":
    main()