# DrugGEN-BACE1-BBB implementation

## Scope
This implementation keeps the supplied DrugGEN graph-transformer GAN and adds only the planned BBB conditioning pathway. It does **not** replace DrugGEN with a VAE, RL generator, diffusion model, or GNN generator.

### Architecture
`ChEMBL graph -> original DrugGEN Generator -> Graph Transformer -> molecule`

with the planned second condition pathway:

`[BACE1 target=1, BBB condition] -> condition projection -> Generator/Discriminator Transformer representations`

The Discriminator receives the same condition so that real target molecules are compared under their observed BBB label and generated molecules are evaluated under the desired BACE1+BBB condition.

## Sources
- DrugGEN official implementation: https://github.com/HUBioDataLab/DrugGEN
- DrugGEN paper: Nature Machine Intelligence, 2025, DOI 10.1038/s42256-025-01082-y.
- B3DB: Scientific Data, 2021, DOI 10.1038/s41597-021-01069-5.
- B3DB code/data: https://github.com/theochem/B3DB
- B3clf reference implementation: https://github.com/theochem/B3clf

## Required data
1. `chembl.smi`: one general drug-like SMILES per line.
2. `bace1_bbb.smi`: BACE1 target molecules, one SMILES per line.
3. `bace1_bbb_labels.csv`: exact columns `smiles,bbb_label`, with `bbb_label` 0/1.

Run `prepare_bace1_bbb.py` to create (2) and (3) from a BACE1 activity table and B3DB table. The script uses exact canonical-SMILES overlap and never invents BBB labels.

## Existing continuous BBB experiment

The saved DrugGEN-BACE1-AffinityBBB-Conditioned-v3 checkpoint uses 7,682 BACE1 molecules from druggen_data/bace1_continuous_bbb.csv. Its BBB condition is predicted_logBB, as recorded in that file and the run configuration. This is proxy conditioning on model-predicted BBB values, not training on measured B3DB labels. In the local source tables, exact canonical-SMILES overlap is only 2 molecules (8,271 unique BACE1 molecules and 1,058 B3DB molecules), which is too small for meaningful paired measured-label training. Interpret the saved checkpoint and its evaluations accordingly.

## Training information
Defaults intentionally follow the supplied DrugGEN baseline: max atoms 45, Transformer dimension 128, depth 1, 8 attention heads, MLP ratio 3, batch size 128, 10 epochs, generator/discriminator LR 1e-5, AdamW betas 0.9/0.999, gradient-penalty lambda 10, dropout 0.

Every run writes `run_config.json` containing source paths, sample counts, architecture settings, epochs, batch size, seed, device, and graph dimensions.

## Run
```bash
python prepare_bace1_bbb.py --bace1 <bace1.csv> --b3db <B3DB.csv>
python train_joint.py --chembl_smi data/chembl.smi --bace1_smi data/bace1_bbb.smi --labels_csv data/bace1_bbb_labels.csv
```

## Important limitation
The paired BACE1+BBB overlap must be present before training. The implementation deliberately fails rather than fabricate BBB labels when a BACE1 molecule has no BBB measurement.
