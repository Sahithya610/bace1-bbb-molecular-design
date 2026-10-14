from pathlib import Path

from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders


# ============================================================
# DrugGEN preprocessing for BACE1 + continuous BBB project
#
# Sources:
#   - Original DrugGEN preprocessing/data pipeline
#   - General ChEMBL molecular dataset: data/chembl_train.smi
#   - BACE1 molecules: druggen_data/bace1.smi
#
# Dataset sizes:
#   - ChEMBL: 1,495,055 SMILES
#   - BACE1: 8,271 unique SMILES
#
# Architecture:
#   Original DrugGEN graph representation
#   max_atom = 45
#
# This step ONLY prepares graph data.
# No model training is performed here.
# ============================================================


DATA_DIR = Path("data")
DRUGGEN_DIR = Path("druggen_data")

CHEMBL_SMI = DATA_DIR / "chembl_train.smi"
BACE1_SMI = DRUGGEN_DIR / "bace1.smi"

MAX_ATOM = 45


def count_smiles(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():

    # --------------------------------------------------------
    # 1. Verify required input files
    # --------------------------------------------------------
    if not CHEMBL_SMI.exists():
        raise FileNotFoundError(f"Missing: {CHEMBL_SMI}")

    if not BACE1_SMI.exists():
        raise FileNotFoundError(f"Missing: {BACE1_SMI}")

    print("=" * 70)
    print("DrugGEN preprocessing")
    print("=" * 70)

    print(f"ChEMBL file : {CHEMBL_SMI}")
    print(f"BACE1 file  : {BACE1_SMI}")
    print(f"MAX_ATOM    : {MAX_ATOM}")

    print()
    print("Input molecule counts:")
    print(f"ChEMBL: {count_smiles(CHEMBL_SMI):,}")
    print(f"BACE1 : {count_smiles(BACE1_SMI):,}")

    # --------------------------------------------------------
    # 2. Build the original DrugGEN encoders/decoders
    #
    # IMPORTANT:
    # The encoders are constructed jointly from the general
    # ChEMBL and BACE1 molecular files, following DrugGEN's
    # original preprocessing approach.
    # --------------------------------------------------------
    print()
    print("Building atom and bond encoders/decoders...")

    atom_encoder, atom_decoder, bond_encoder, bond_decoder = \
        get_encoders_decoders(
            str(CHEMBL_SMI),
            str(BACE1_SMI),
            MAX_ATOM
        )

    print(f"Number of atom types : {len(atom_decoder)}")
    print(f"Number of bond types : {len(bond_decoder)}")

    # --------------------------------------------------------
    # 3. Preprocess general ChEMBL dataset
    # --------------------------------------------------------
    print()
    print("-" * 70)
    print("Processing general ChEMBL dataset")
    print("-" * 70)

    chembl_file = CHEMBL_SMI.stem + str(MAX_ATOM) + ".pt"

    chembl_dataset = DruggenDataset(
        str(DATA_DIR),
        chembl_file,
        str(CHEMBL_SMI),
        MAX_ATOM,
        False,
        atom_encoder,
        atom_decoder,
        bond_encoder,
        bond_decoder,
    )

    print(f"ChEMBL processed successfully.")
    print(f"Graph count : {len(chembl_dataset)}")

    # --------------------------------------------------------
    # 4. Preprocess BACE1 dataset
    # --------------------------------------------------------
    print()
    print("-" * 70)
    print("Processing BACE1 dataset")
    print("-" * 70)

    bace1_file = BACE1_SMI.stem + str(MAX_ATOM) + ".pt"

    bace1_dataset = DruggenDataset(
        str(DRUGGEN_DIR),
        bace1_file,
        str(BACE1_SMI),
        MAX_ATOM,
        False,
        atom_encoder,
        atom_decoder,
        bond_encoder,
        bond_decoder,
    )

    print(f"BACE1 processed successfully.")
    print(f"Graph count : {len(bace1_dataset)}")

    # --------------------------------------------------------
    # 5. Verify graph dimensions
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(f"ChEMBL graphs : {len(chembl_dataset):,}")
    print(f"BACE1 graphs  : {len(bace1_dataset):,}")

    print(f"Atom features : {chembl_dataset[0].x.shape}")
    print(f"Atom/bond data: {chembl_dataset[0].edge_index.shape}")

    print()
    print("The molecular graph data is ready for the DrugGEN training stage.")
    print("Continuous BBB labels remain in:")
    print("druggen_data/bace1_continuous_bbb.csv")


if __name__ == "__main__":
    main()