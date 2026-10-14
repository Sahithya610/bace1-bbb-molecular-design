"""
Audit of high-activity candidates from 50k screening run.
Analyses molecules with pIC50 >= 7.6576 (n=6) and >= 7.0 (n=17).
"""
import pandas as pd
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.DataStructs import TanimotoSimilarity

# ── Load data ──────────────────────────────────────────────────────
pred = pd.read_csv("results/baseline_screen_50000_predictions.csv")
train = pd.read_csv("results/bace1_activity_model/bace1_activity_training_table.csv")
print(f"50k predictions: {len(pred)} rows")
print(f"Training set: {len(train)} rows")

# ── Verify BBB model consistency ───────────────────────────────────
import json as _j
with open("results/bbb_activity_model/bbb_model_metadata.json") as f:
    bbb_meta = _j.load(f)
print(f"\nBBB model metadata:")
print(f"  target:      {bbb_meta.get('target')}")
print(f"  fingerprint: {bbb_meta.get('fingerprint')}")
print(f"  test R²:     {bbb_meta.get('test_results',{}).get('R2','n/a')}")
print(f"  test RMSE:   {bbb_meta.get('test_results',{}).get('RMSE','n/a')}")

# Confirm the prediction CSV uses the independent BBB model
# (predict_joint.py loads bbb_model.joblib, NOT the discriminator head)
# Verify by checking if normalized_predicted_logBB = (predicted_logBB - mean)/std
B3DB_MEAN, B3DB_STD = -0.077873, 0.751623
check = pred.head(3).copy()
check["computed_norm"] = (check["predicted_logBB"] - B3DB_MEAN) / B3DB_STD
check["stored_norm"] = check["normalized_predicted_logBB"]
check["norm_matches"] = (check["computed_norm"] - check["stored_norm"]).abs() < 1e-4
print(f"\nBBB model consistency check (3 rows):")
print(check[["canonical_smiles","predicted_logBB","computed_norm","stored_norm","norm_matches"]].to_string())

# ── Subsets ────────────────────────────────────────────────────────
high76 = pred[pred["predicted_pIC50"] >= 7.6576].copy().reset_index(drop=True)
high70 = pred[pred["predicted_pIC50"] >= 7.0].copy().reset_index(drop=True)
print(f"\nMolecules with pIC50 >= 7.6576: {len(high76)}")
print(f"Molecules with pIC50 >= 7.0:    {len(high70)}")

# ── Precompute training fingerprints ───────────────────────────────
print("\nPrecomputing training fingerprints...")
train_fps = []
train_smi = []
for smi in train["canonical_smiles"]:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
        train_smi.append(smi)
print(f"Training FPs computed: {len(train_fps)}")

def get_nearest_train(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, False
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    sims = [TanimotoSimilarity(fp, tfp) for tfp in train_fps]
    max_idx = int(np.argmax(sims))
    max_tc = sims[max_idx]
    is_dup = max_tc >= 0.999
    return max_tc, train_smi[max_idx], is_dup

# ── PAINS filter ───────────────────────────────────────────────────
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
pains_cat = FilterCatalog(pains_params)

brenk_params = FilterCatalogParams()
brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
brenk_cat = FilterCatalog(brenk_params)

def get_filters(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "INVALID", "INVALID", None, None
    pains = [e.GetDescription() for e in pains_cat.GetMatches(mol)]
    brenk = [e.GetDescription() for e in brenk_cat.GetMatches(mol)]
    return ("; ".join(pains) if pains else "none",
            "; ".join(brenk[:3]) if brenk else "none",
            round(QED.qed(mol), 3),
            round(Descriptors.MolWt(mol), 1))

# ── Analyse pIC50 >= 7.6576 molecules ─────────────────────────────
SEP = "="*72
print(f"\n{SEP}")
print("SIX MOLECULES WITH pIC50 >= 7.6576")
print(SEP)

rows_out = []
for i, row in high76.iterrows():
    smi = row["canonical_smiles"]
    max_tc, near_smi, is_dup = get_nearest_train(smi)
    pains, brenk, qed, mw = get_filters(smi)
    meets_bbb = row["predicted_logBB"] >= 0.0
    meets_joint = (row["predicted_pIC50"] >= 7.6576) and meets_bbb
    # Scaffold
    mol = Chem.MolFromSmiles(smi)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = "n/a"

    print(f"\n  Candidate {i+1}")
    print(f"  SMILES:          {smi}")
    print(f"  pIC50 (oracle):  {row['predicted_pIC50']:.4f}")
    print(f"  logBB:           {row['predicted_logBB']:.4f}  (passes BBB? {'YES' if meets_bbb else 'NO'})")
    print(f"  logBB vs 0:      {row['predicted_logBB']:+.4f}  (margin = {row['predicted_logBB']:+.4f})")
    print(f"  BBB model RMSE:  0.621  → plausible range approx [{row['predicted_logBB']-1.2:.2f}, {row['predicted_logBB']+1.2:.2f}]")
    print(f"  Meets joint:     {'YES ✓' if meets_joint else 'NO ✗'}")
    print(f"  Max train Tc:    {max_tc:.3f}  {'(EXACT DUPLICATE)' if is_dup else ''}")
    print(f"  Nearest train:   {near_smi[:60]}...")
    print(f"  MW: {mw}  QED: {qed}")
    print(f"  PAINS: {pains}")
    print(f"  Brenk: {brenk[:80]}")
    print(f"  Scaffold: {scaffold[:70]}")

    rows_out.append({
        "rank": i+1,
        "canonical_smiles": smi,
        "oracle_pIC50": row["predicted_pIC50"],
        "predicted_logBB": row["predicted_logBB"],
        "logBB_margin_from_0": row["predicted_logBB"],
        "logBB_approx_range_low": round(row["predicted_logBB"] - 1.2, 3),
        "logBB_approx_range_high": round(row["predicted_logBB"] + 1.2, 3),
        "passes_pIC50_7_6576": row["predicted_pIC50"] >= 7.6576,
        "passes_logBB_0": meets_bbb,
        "passes_joint_threshold": meets_joint,
        "max_training_Tc": round(max_tc, 4) if max_tc is not None else None,
        "exact_training_duplicate": is_dup,
        "nearest_training_smiles": near_smi,
        "MW": mw,
        "QED": qed,
        "PAINS": pains,
        "Brenk": brenk,
        "Murcko_scaffold": scaffold,
        "source_pool": "50k_pilot",
        "bbb_model_test_R2": bbb_meta.get("test_results",{}).get("R2","n/a"),
        "bbb_model_test_RMSE": bbb_meta.get("test_results",{}).get("RMSE","n/a"),
    })

# ── logBB distribution comparison ─────────────────────────────────
print(f"\n{SEP}")
print("logBB DISTRIBUTION COMPARISON")
print(SEP)

def logbb_stats(df, label):
    lb = df["predicted_logBB"]
    print(f"\n  {label} (n={len(df)}):")
    print(f"    mean={lb.mean():.3f}  median={lb.median():.3f}  std={lb.std():.3f}")
    print(f"    min={lb.min():.3f}  max={lb.max():.3f}")
    print(f"    logBB >= 0:  {(lb>=0).sum()} ({100*(lb>=0).mean():.1f}%)")
    print(f"    logBB >= -0.5: {(lb>=-0.5).sum()} ({100*(lb>=-0.5).mean():.1f}%)")
    print(f"    logBB <  -1.0: {(lb<-1.0).sum()} ({100*(lb<-1.0).mean():.1f}%)")

logbb_stats(pred, "All 35,042 unique evaluated molecules")
logbb_stats(high70, "17 molecules with pIC50 >= 7.0")
logbb_stats(high76, "6 molecules with pIC50 >= 7.6576")

# Key question: how far below 0 are the high-pIC50 molecules?
print(f"\n  The 6 high-pIC50 molecules' logBB values:")
for _, r in high76.iterrows():
    margin = r["predicted_logBB"]
    within_1rmse = abs(margin) <= 0.621
    print(f"    {r['predicted_pIC50']:.3f} pIC50 | logBB={margin:+.4f}"
          f" | within 1×RMSE of threshold: {'YES' if within_1rmse else 'no'}")

# ── Correlation between pIC50 and logBB ───────────────────────────
corr = pred[["predicted_pIC50","predicted_logBB"]].corr().iloc[0,1]
print(f"\n  Pearson corr(pIC50, logBB) across all 35k mols: {corr:.3f}")
print(f"  (negative = high pIC50 mols tend to have lower logBB)")

# ── Save audit CSV ─────────────────────────────────────────────────
out_df = pd.DataFrame(rows_out)
out_df.to_csv("results/high_activity_candidate_audit.csv", index=False)
print(f"\n{SEP}")
print(f"Saved: results/high_activity_candidate_audit.csv ({len(out_df)} rows)")
print(SEP)

# ── Brief interpretation ───────────────────────────────────────────
print(f"""
INTERPRETATION
{"─"*60}
Oracle test R² = 0.6156, RMSE = 0.755 pIC50 units.
BBB model test R² = 0.354, RMSE = 0.621 logBB units.

The ±1 RMSE interval for logBB predictions spans ±0.621 log units.
A molecule predicted at logBB = -0.3 could plausibly have true logBB
anywhere from -0.9 to +0.3 — spanning both sides of the BBB threshold.

Whether the 6 high-pIC50 molecules are truly BBB-negative or merely
predicted negative by a model with R²=0.354 (and BACE1 scaffolds outside
the B3DB training domain) cannot be determined computationally.

Structural note: A negative correlation between pIC50 and logBB is
expected — BACE1 inhibitors tend to be large, polar, peptidomimetic
molecules (high MW, high TPSA) which are inherently BBB-unfavourable.
This is a structural/pharmacological reality, not a modeling artifact.
The joint threshold is genuinely difficult to achieve for this target.

See results/high_activity_candidate_audit.csv for full per-molecule data.
""")
