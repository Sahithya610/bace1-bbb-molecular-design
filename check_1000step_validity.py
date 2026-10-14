import pandas as pd

for tag, f in [("cond55", "results/vg_1000step_cond55.csv"),
               ("cond80", "results/vg_1000step_cond80.csv")]:
    df = pd.read_csv(f)
    n_total = len(df)
    n_valid = df["valid"].sum()
    # Check for degenerate molecules (single atom)
    if "smiles" in df.columns:
        non_empty = df[df["smiles"].notna() & (df["smiles"] != "")]
        tiny = non_empty[non_empty["smiles"].str.len() <= 3]
        real = non_empty[non_empty["smiles"].str.len() > 3]
        print(f"{tag}: {n_valid}/{n_total} valid | "
              f"tiny (len≤3): {len(tiny)} | real chemistry (len>3): {len(real)}")
        print(f"  atom count dist: min={df['decoded_atom_count_before_component_filter'].min():.0f} "
              f"mean={df['decoded_atom_count_before_component_filter'].mean():.1f} "
              f"max={df['decoded_atom_count_before_component_filter'].max():.0f}")
        if len(real) > 0:
            print(f"  Sample real SMILES: {real['smiles'].iloc[0]}")
