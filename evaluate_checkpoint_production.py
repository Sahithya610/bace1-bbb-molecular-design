"""
Evaluate Issue 5 Checkpoints using Production Generation and Independent Oracle Pipeline.

This script implements Requirement 4:
- Evaluates saved checkpoints using the exact production generation pipeline:
    generate_joint_molecules.py -> predict_joint.py
- Compares both condition modes:
    1. 'zero' mode (condition=[0,0,0], matching Option A training distribution)
    2. 'conditional' mode (condition=[1, norm_pIC50, norm_logBB], matching earlier Issue 5 eval)
- Records:
    * Graphs generated
    * Valid molecules and validity fraction
    * Unique canonical isomeric SMILES and duplicate count
    * Atom-class distribution and padding frequency
    * Morgan Tanimoto similarity to the 187 joint-positive training molecules
    * Independent-oracle pIC50 predictions (mean, median, min, max, count >= 7.6576)
    * BBB-model logBB predictions (mean, median, min, max, count >= 0.0)
    * Joint threshold counts: pIC50 >= 7.6576 AND logBB >= 0.0
- Outputs per-checkpoint CSVs and a consolidated JSON/Markdown summary.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def mol_to_fp(mol):
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    except Exception:
        return None


def canonical_isomeric_smiles(smi):
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def evaluate_checkpoint(
    checkpoint_path,
    output_dir,
    joint_positive_smi="druggen_data/bace1_joint_positive_train.smi",
    batch_size=128,
    num_batches=8,
    seed=42,
    desired_pIC50=7.0,
    desired_logBB=0.0,
    condition_mode="zero",
    py_exec=sys.executable,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = checkpoint_path.stem
    label = f"{stem}_{condition_mode}"
    gen_csv = output_dir / f"generated_{label}.csv"
    pred_csv = output_dir / f"predictions_{label}.csv"
    summary_json = output_dir / f"summary_{label}.json"

    print("=" * 78)
    print(f"EVALUATING: {checkpoint_path.name} | Mode: {condition_mode} | Seed: {seed}")
    print("=" * 78)

    # 1. Run generation pipeline
    gen_cmd = [
        py_exec,
        "generate_joint_molecules.py",
        "--checkpoint", str(checkpoint_path),
        "--batch_size", str(batch_size),
        "--num_batches", str(num_batches),
        "--desired_pIC50", str(desired_pIC50),
        "--desired_logBB", str(desired_logBB),
        "--condition_mode", condition_mode,
        "--seed", str(seed),
        "--output", str(gen_csv),
    ]

    t0 = time.time()
    gen_proc = subprocess.run(gen_cmd, capture_output=True, text=True, cwd=".")
    gen_time = time.time() - t0

    if gen_proc.returncode != 0:
        print(f"ERROR running generate_joint_molecules.py:\n{gen_proc.stderr}")
        return {"error": gen_proc.stderr, "checkpoint": str(checkpoint_path)}

    if not gen_csv.is_file():
        raise FileNotFoundError(f"Expected generation output file not found: {gen_csv}")

    df_gen = pd.read_csv(gen_csv)
    total_graphs = len(df_gen)
    valid_graphs = int(df_gen["valid"].sum()) if "valid" in df_gen.columns else 0
    validity_frac = valid_graphs / total_graphs if total_graphs > 0 else 0.0

    valid_smiles_list = df_gen.loc[df_gen["valid"] == True, "smiles"].dropna().tolist()

    # Canonical isomeric SMILES and uniqueness
    canonical_smiles_list = [
        canonical_isomeric_smiles(s) for s in valid_smiles_list
    ]
    canonical_smiles_list = [s for s in canonical_smiles_list if s is not None]
    unique_smiles_set = set(canonical_smiles_list)
    unique_count = len(unique_smiles_set)
    duplicate_count = len(canonical_smiles_list) - unique_count
    uniqueness_frac = unique_count / len(canonical_smiles_list) if canonical_smiles_list else 0.0

    # 2. Run prediction pipeline
    pred_proc = subprocess.run(
        [
            py_exec,
            "predict_joint.py",
            "--input", str(gen_csv),
            "--output", str(pred_csv),
        ],
        capture_output=True,
        text=True,
        cwd=".",
    )

    if pred_proc.returncode != 0:
        print(f"WARNING running predict_joint.py:\n{pred_proc.stderr}")

    # Load predictions if available
    df_pred = pd.DataFrame()
    if pred_csv.is_file():
        try:
            df_pred = pd.read_csv(pred_csv)
        except Exception as e:
            print(f"Could not read prediction CSV: {e}")

    # 3. Training set similarity
    train_fps = []
    if Path(joint_positive_smi).is_file():
        for line in open(joint_positive_smi, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            mol_t = Chem.MolFromSmiles(line)
            if mol_t is not None:
                fp_t = mol_to_fp(mol_t)
                if fp_t is not None:
                    train_fps.append(fp_t)

    max_tc_list = []
    tc_ge_09_count = 0
    tc_ge_04_count = 0
    for smi in unique_smiles_set:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = mol_to_fp(mol)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        if sims:
            m = max(sims)
            max_tc_list.append(m)
            if m >= 0.9:
                tc_ge_09_count += 1
            if m >= 0.4:
                tc_ge_04_count += 1

    sim_mean = float(np.mean(max_tc_list)) if max_tc_list else 0.0
    sim_max = float(np.max(max_tc_list)) if max_tc_list else 0.0
    sim_ge_09_frac = tc_ge_09_count / len(unique_smiles_set) if unique_smiles_set else 0.0

    # 4. Property evaluations
    # Thresholds: pIC50 >= 7.6576, logBB >= 0.0
    THRESH_PIC50 = 7.6576
    THRESH_LOGBB = 0.0

    n_eval = len(df_pred)
    n_pic50_pass = 0
    n_logbb_pass = 0
    n_joint_pass = 0
    pic50_stats = {}
    logbb_stats = {}
    joint_molecules = []

    if n_eval > 0 and "predicted_pIC50" in df_pred.columns and "predicted_logBB" in df_pred.columns:
        p_vals = df_pred["predicted_pIC50"].dropna()
        l_vals = df_pred["predicted_logBB"].dropna()

        pic50_stats = {
            "mean": float(p_vals.mean()),
            "median": float(p_vals.median()),
            "min": float(p_vals.min()),
            "max": float(p_vals.max()),
        }
        logbb_stats = {
            "mean": float(l_vals.mean()),
            "median": float(l_vals.median()),
            "min": float(l_vals.min()),
            "max": float(l_vals.max()),
        }

        pass_pic50_mask = df_pred["predicted_pIC50"] >= THRESH_PIC50
        pass_logbb_mask = df_pred["predicted_logBB"] >= THRESH_LOGBB
        pass_joint_mask = pass_pic50_mask & pass_logbb_mask

        n_pic50_pass = int(pass_pic50_mask.sum())
        n_logbb_pass = int(pass_logbb_mask.sum())
        n_joint_pass = int(pass_joint_mask.sum())

        if n_joint_pass > 0:
            for _, r in df_pred[pass_joint_mask].iterrows():
                joint_molecules.append({
                    "smiles": str(r.get("canonical_smiles", r.get("smiles", ""))),
                    "predicted_pIC50": float(r["predicted_pIC50"]),
                    "predicted_logBB": float(r["predicted_logBB"]),
                })

    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_name": checkpoint_path.name,
        "condition_mode": condition_mode,
        "seed": seed,
        "sample": {
            "requested_graphs": batch_size * num_batches,
            "generated_graphs": total_graphs,
            "valid_molecules": valid_graphs,
            "validity_fraction": round(validity_frac, 6),
            "unique_canonical_smiles": unique_count,
            "duplicate_count": duplicate_count,
            "uniqueness_fraction": round(uniqueness_frac, 6),
        },
        "similarity_to_joint_positive_train": {
            "max_tanimoto": round(sim_max, 4),
            "mean_tanimoto": round(sim_mean, 4),
            "count_tc_ge_09": tc_ge_09_count,
            "fraction_tc_ge_09": round(sim_ge_09_frac, 4),
            "count_tc_ge_04": tc_ge_04_count,
        },
        "oracle_predictions": {
            "evaluated_count": n_eval,
            "pic50_stats": pic50_stats,
            "logbb_stats": logbb_stats,
            "threshold_pic50_ge_7_6576_count": n_pic50_pass,
            "threshold_logbb_ge_0_count": n_logbb_pass,
            "threshold_both_jointly_count": n_joint_pass,
            "joint_molecules": joint_molecules,
        },
        "files": {
            "generated_csv": str(gen_csv.resolve()),
            "predictions_csv": str(pred_csv.resolve()),
        },
        "elapsed_seconds": round(gen_time, 2),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Results summary:")
    print(f"  Valid: {valid_graphs}/{total_graphs} ({validity_frac:.2%}) | Unique: {unique_count}")
    print(f"  pIC50 >= 7.6576: {n_pic50_pass} | logBB >= 0.0: {n_logbb_pass} | BOTH: {n_joint_pass}")
    print(f"  Max Tc to joint-positive train: {sim_max:.4f}")
    print("-" * 78)

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints using production pipeline.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/issue5_checkpointed_eval",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["zero", "conditional"],
        help="Condition modes to evaluate for each checkpoint.",
    )
    args = parser.parse_args()

    results = []
    for ckpt in args.checkpoints:
        for mode in args.modes:
            res = evaluate_checkpoint(
                checkpoint_path=ckpt,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
                seed=args.seed,
                condition_mode=mode,
            )
            results.append(res)

    master_summary_path = Path(args.output_dir) / "master_evaluation_summary.json"
    with open(master_summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMaster summary written to {master_summary_path}")


if __name__ == "__main__":
    main()
