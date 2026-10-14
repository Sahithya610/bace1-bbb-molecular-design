"""Parse already-computed Vina pose files and build final results table."""
import os, glob
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV = "results/top200_scaffold_analysis.csv"
POSES_DIR = "docking/poses"
OUTPUT_CSV = "results/docking/bace1_druggen_top_vina.csv"

df = pd.read_csv(INPUT_CSV)
df = df[df["Ro5"] == 1].sort_values("predicted_pIC50", ascending=False).head(50).reset_index(drop=True)

results = []
for i, row in df.iterrows():
    rank  = i + 1
    pose_f = f"{POSES_DIR}/pose_{rank:03d}.pdbqt"
    score = None
    if os.path.exists(pose_f):
        with open(pose_f) as f:
            for line in f:
                if "VINA RESULT" in line:
                    try:
                        score = float(line.split(":")[1].split()[0])
                        break
                    except Exception:
                        pass
    results.append({
        "rank": rank,
        "canonical_smiles": row["canonical_smiles"],
        "oracle_pIC50": row["predicted_pIC50"],
        "vina_score": score,
        "predicted_logBB": row["predicted_logBB"],
        "MW": row["MW"], "LogP": row["LogP"],
        "TPSA": row["TPSA"], "HBD": row["HBD"],
        "status": "ok" if score is not None else "missing",
    })

res = pd.DataFrame(results)
res.to_csv(OUTPUT_CSV, index=False)

docked = res[res["vina_score"].notna()].sort_values("vina_score")
print(f"Parsed {len(docked)}/{len(df)} pose files\n")
print(f"Best Vina:  {docked['vina_score'].min():.2f} kcal/mol")
print(f"Mean Vina:  {docked['vina_score'].mean():.2f} kcal/mol")

print(f"\n  {'#':>3}  {'Vina':>6}  {'Oracle':>7}  {'logBB':>6}  {'MW':>6}  SMILES")
print("  " + "-"*85)
for j, (_, r) in enumerate(docked.head(15).iterrows(), 1):
    s = r["canonical_smiles"]
    s = s[:52]+"..." if len(s)>52 else s
    print(f"  {j:3d}  {r['vina_score']:+6.2f}  "
          f"{r['oracle_pIC50']:7.3f}  "
          f"{r['predicted_logBB']:+6.3f}  "
          f"{r['MW']:6.1f}  {s}")

print(f"\nSaved: {OUTPUT_CSV}")
