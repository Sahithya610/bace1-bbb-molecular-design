r"""
analyze_readout_condition_pathway.py  (READ-ONLY, weights only, no data needed)

    python analyze_readout_condition_pathway.py
    python analyze_readout_condition_pathway.py --ckpt experiments\models\Issue3-ChemistryReadout\joint_generator.pt

In ConditionalGenerator the final layers are
    node_logits = readout_n( concat[ node_repr, readout_condition(c) ] )
    edge_logits = readout_e( concat[ edge_repr, readout_condition(c) ] )
Because the concat is followed by ONE linear layer, the condition adds
    W_cond @ tanh(W_rc @ c + b_rc)
to the logits, and that term is identical at every node (or edge) position of a
graph. This script computes that position-independent per-class logit shift for
several requested pIC50 values relative to a reference pIC50.

What it does NOT include: the separate condition injection before the
Transformer (needs real graphs to evaluate). A decoded atom or bond changes
only if a class-score gap at some position is smaller than the shift, so the
numbers here must be compared with real logit margins from a data-based test.
Nothing is written except results\step5a_readout_pathway_report.txt.
"""
import argparse
import os

import numpy as np

PIC50_MEAN, PIC50_STD = 6.691315, 1.308678
BBB_MEAN, BBB_STD = -0.077873, 0.751623

# Labels taken from the mapping report (class -> atomic number / bond type).
NODE_LABELS = {0: "pad(Z0)", 1: "H", 2: "B", 3: "C", 4: "N", 5: "O", 6: "F",
               7: "Si", 8: "P", 9: "S", 10: "Cl", 11: "Se", 12: "Br", 13: "I"}
EDGE_LABELS = {0: "no-bond", 1: "single", 2: "double", 3: "triple", 4: "aromatic"}


def condition(pic50, logbb=0.0, target=1.0):
    return np.array([target,
                     (pic50 - PIC50_MEAN) / PIC50_STD,
                     (logbb - BBB_MEAN) / BBB_STD], dtype=np.float64)


def compute_shifts(w, pic50_values, ref_pic50, logbb=0.0):
    """w: dict of float numpy arrays. Returns {pic50: (node_shift, edge_shift)}
    where each shift is (logit at pic50) - (logit at ref_pic50), per class."""
    dim = w["readout_condition.0.weight"].shape[0]
    wn = w["readout_n.weight"][:, dim:].astype(np.float64)   # [nodes, dim]
    we = w["readout_e.weight"][:, dim:].astype(np.float64)   # [edges, dim]
    wrc = w["readout_condition.0.weight"].astype(np.float64)
    brc = w["readout_condition.0.bias"].astype(np.float64)

    def term(c):
        h = np.tanh(wrc @ c + brc)
        return wn @ h, we @ h

    n_ref, e_ref = term(condition(ref_pic50, logbb))
    out = {}
    for p in pic50_values:
        n, e = term(condition(p, logbb))
        out[p] = (n - n_ref, e - e_ref)
    return out


def format_report(shifts, ref, n_classes, e_classes):
    lines = [f"Position-independent logit shift from readout_condition, relative to pIC50={ref}",
             "(only the pIC50 component of the condition varies; target=1.0, logBB fixed)\n"]
    cols = list(shifts)
    for name, idx, labels, n in (("NODE (atom) classes", 0, NODE_LABELS, n_classes),
                                 ("EDGE (bond) classes", 1, EDGE_LABELS, e_classes)):
        lines.append(name)
        lines.append("  class".ljust(16) + "".join(f"pIC50={c:<6}".rjust(14) for c in cols))
        for k in range(n):
            lab = labels.get(k, str(k)) if n == len(labels) else str(k)
            lines.append(f"  {k:2d} {lab:<10}" + "".join(f"{shifts[c][idx][k]:+14.4f}" for c in cols))
        lines.append("  max |shift|".ljust(16) + "".join(f"{np.abs(shifts[c][idx]).max():14.4f}" for c in cols))
        lines.append("  spread (max-min over classes; only class DIFFERENCES can change an argmax)".ljust(16))
        lines.append("".ljust(16) + "".join(f"{np.ptp(shifts[c][idx]):14.4f}" for c in cols))
        lines.append("")
    return "\n".join(lines)


def load_state_dict(path):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "generator", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    return {k: v.detach().cpu().numpy() for k, v in sd.items() if hasattr(v, "detach")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("experiments", "models", "Issue3-ChemistryReadout", "joint_generator.pt"))
    ap.add_argument("--ref", type=float, default=5.5, help="reference pIC50 (default 5.5)")
    ap.add_argument("--pic50", type=float, nargs="+", default=[6.5, 7.0, 8.0, 9.0])
    ap.add_argument("--logbb", type=float, default=0.0)
    args = ap.parse_args()

    w = load_state_dict(args.ckpt)
    need = ["readout_condition.0.weight", "readout_condition.0.bias", "readout_n.weight", "readout_e.weight"]
    missing = [k for k in need if k not in w]
    if missing:
        raise SystemExit(f"{args.ckpt} lacks {missing}; it is not a ChemistryReadout generator. "
                         "(The Issue3-Diagnostic checkpoint has no readout_condition.)")
    print(f"Checkpoint: {args.ckpt}")
    shifts = compute_shifts(w, args.pic50, args.ref, args.logbb)
    report = format_report(shifts, args.ref, w["readout_n.weight"].shape[0], w["readout_e.weight"].shape[0])
    print(report)

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "step5a_readout_pathway_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {args.ckpt}\n" + report)
    print(f"Saved report to {out}")


if __name__ == "__main__":
    main()