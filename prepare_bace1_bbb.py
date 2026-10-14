"""Prepare the paired BACE1 + BBB dataset.

Inputs:
  --bace1 : CSV with at least a smiles column. Optional activity columns are retained.
  --b3db  : B3DB CSV/TSV with SMILES and BBB+/BBB- (or equivalent) column.
Outputs:
  data/bace1_bbb.smi
  data/bace1_bbb_labels.csv

Exact canonical-SMILES overlap is used. No generated or imputed BBB labels are created.
"""
import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem


def read_table(path):
    return pd.read_csv(path, sep=None, engine="python")


def find_col(df, names):
    lower={c.lower().strip():c for c in df.columns}
    for n in names:
        if n in lower: return lower[n]
    for c in df.columns:
        lc=c.lower().strip()
        if any(n in lc for n in names): return c
    raise ValueError(f"Could not find any of {names}. Columns: {list(df.columns)}")


def canon(s):
    m=Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None

p=argparse.ArgumentParser()
p.add_argument('--bace1', required=True)
p.add_argument('--b3db', required=True)
p.add_argument('--out_dir', default='data')
a=p.parse_args()

b=read_table(a.bace1); bb=read_table(a.b3db)
bsm=find_col(b,['smiles','canonical_smiles','isomeric_smiles'])
bbsm=find_col(bb,['smiles','canonical_smiles','isomeric_smiles'])
bbcol=find_col(bb,['bbb+/bbb-','bbb','bbb_label','permeability'])

b['canonical_smiles']=b[bsm].map(canon)
bb['canonical_smiles']=bb[bbsm].map(canon)
b= b.dropna(subset=['canonical_smiles']).drop_duplicates('canonical_smiles')
bb=bb.dropna(subset=['canonical_smiles']).drop_duplicates('canonical_smiles')

def label(x):
    s=str(x).strip().lower()
    if s in {'bbb+','bbb positive','positive','1','yes','true'}: return 1
    if s in {'bbb-','bbb negative','negative','0','no','false'}: return 0
    return None
bb['bbb_label']=bb[bbcol].map(label)
bb=bb.dropna(subset=['bbb_label'])
out=b[['canonical_smiles']].merge(bb[['canonical_smiles','bbb_label']], on='canonical_smiles', how='inner')
out=out.drop_duplicates('canonical_smiles')
if out.empty: raise SystemExit('No exact canonical-SMILES overlap between BACE1 and B3DB.')
Path(a.out_dir).mkdir(parents=True,exist_ok=True)
out.rename(columns={'canonical_smiles':'smiles'}).to_csv(Path(a.out_dir)/'bace1_bbb_labels.csv',index=False)
out['smiles'].to_csv(Path(a.out_dir)/'bace1_bbb.smi',index=False,header=False)
print(f'BACE1/BBB paired samples: {len(out)}')
print(out.bbb_label.value_counts().sort_index().to_dict())
