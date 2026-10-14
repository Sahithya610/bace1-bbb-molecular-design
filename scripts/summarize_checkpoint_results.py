import glob
import json
from pathlib import Path
import pandas as pd

summary_file = Path("results/issue5_checkpointed_eval/master_evaluation_summary.json")
with open(summary_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print("PER-CHECKPOINT EVALUATION TABLE (1,024 graphs per evaluation, Seed 42)")
print("=" * 80)
print("| Checkpoint | Mode | Gen | Valid | Valid% | Unique | Mean pIC50 | Max pIC50 | Mean logBB | Max logBB | logBB>=0 | pIC50>=7.6576 | BOTH | MaxTc |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

for r in data:
    name = r["checkpoint_name"].replace("generator_", "").replace(".pt", "")
    mode = r["condition_mode"]
    s = r["sample"]
    o = r["oracle_predictions"]
    sim = r["similarity_to_joint_positive_train"]
    p_stat = o.get("pic50_stats", {})
    l_stat = o.get("logbb_stats", {})
    mean_p = f"{p_stat.get('mean', 0.0):.3f}" if p_stat else "N/A"
    max_p = f"{p_stat.get('max', 0.0):.3f}" if p_stat else "N/A"
    mean_l = f"{l_stat.get('mean', 0.0):.3f}" if l_stat else "N/A"
    max_l = f"{l_stat.get('max', 0.0):.3f}" if l_stat else "N/A"
    n_p = o.get("threshold_pic50_ge_7_6576_count", 0)
    n_l = o.get("threshold_logbb_ge_0_count", 0)
    n_both = o.get("threshold_both_jointly_count", 0)
    max_tc = f"{sim.get('max_tanimoto', 0.0):.3f}"
    print(f"| {name} | {mode} | {s['generated_graphs']} | {s['valid_molecules']} | {s['validity_fraction']*100:.1f}% | {s['unique_canonical_smiles']} | {mean_p} | {max_p} | {mean_l} | {max_l} | {n_l} | {n_p} | {n_both} | {max_tc} |")

print("\n" + "=" * 80)
print("TOP 10 MOLECULES BY PREDICTED pIC50 (with logBB >= 0.0)")
print("=" * 80)

csvs = glob.glob("results/issue5_checkpointed_eval/predictions_*.csv")
dfs = [pd.read_csv(f) for f in csvs]
all_df = pd.concat(dfs, ignore_index=True)
all_df = all_df.dropna(subset=["canonical_smiles"]).drop_duplicates(subset=["canonical_smiles"])
print(f"Total unique evaluated molecules across all checkpoints: {len(all_df)}")

pos_bbb = all_df[all_df["predicted_logBB"] >= 0.0].sort_values(by="predicted_pIC50", ascending=False)
print(f"Total unique molecules with logBB >= 0.0: {len(pos_bbb)}")

for rank, (_, r) in enumerate(pos_bbb.head(10).iterrows(), 1):
    smi = r["canonical_smiles"]
    p = r["predicted_pIC50"]
    l = r["predicted_logBB"]
    print(f"{rank:>2}. pIC50={p:.4f} | logBB={l:+.4f} | {smi}")
