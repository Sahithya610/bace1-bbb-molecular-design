
"""
Conditional DrugGEN dataset for joint BACE1 affinity + BBB generation.

Condition vector:
    [target_condition, normalized_pIC50, normalized_logBB]

Target condition:
    BACE1 = 1.0

pIC50:
    Continuous BACE1 activity condition.

    Normalization:
        mean = 6.691315
        std  = 1.308678

BBB:
    Continuous predicted logBB condition.

    Normalization:
        mean = -0.077873
        std  = 0.751623
"""

import pandas as pd
import torch
from rdkit import Chem

from src.data.dataset import DruggenDataset


# ============================================================
# Normalization constants
# ============================================================

PIC50_MEAN = 6.691315
PIC50_STD = 1.308678

BBB_MEAN = -0.077873
BBB_STD = 0.751623


# ============================================================
# SMILES canonicalization
# ============================================================

def canonicalize_smiles(smiles):
    """
    Convert a valid SMILES string into RDKit canonical SMILES.

    Returns:
        canonical SMILES string, or None if invalid.
    """

    if smiles is None:
        return None

    smiles = str(smiles).strip()

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
    )


# ============================================================
# Conditional dataset
# ============================================================

class ConditionalDruggenDataset(DruggenDataset):

    def __init__(
        self,
        root,
        dataset_file,
        raw_files,
        max_atom,
        features,
        atom_encoder,
        atom_decoder,
        bond_encoder,
        bond_decoder,
        labels_csv,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):

        # ----------------------------------------------------
        # Load the original DrugGEN graph dataset
        # ----------------------------------------------------
        super().__init__(
            root=root,
            dataset_file=dataset_file,
            raw_files=raw_files,
            max_atom=max_atom,
            features=features,
            atom_encoder=atom_encoder,
            atom_decoder=atom_decoder,
            bond_encoder=bond_encoder,
            bond_decoder=bond_decoder,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        # ----------------------------------------------------
        # Load condition table
        # ----------------------------------------------------
        labels = pd.read_csv(
            labels_csv
        )

        required_columns = {
            "canonical_smiles",
            "pIC50",
            "predicted_logBB",
        }

        missing = (
            required_columns
            - set(labels.columns)
        )

        if missing:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing)}"
            )

        # ----------------------------------------------------
        # Validate numerical labels
        # ----------------------------------------------------
        if labels["pIC50"].isna().any():
            raise ValueError(
                "Missing pIC50 values found."
            )

        if labels["predicted_logBB"].isna().any():
            raise ValueError(
                "Missing predicted_logBB values found."
            )

        # ----------------------------------------------------
        # Normalize pIC50
        # ----------------------------------------------------
        labels["normalized_pIC50"] = (
            labels["pIC50"].astype(float)
            - PIC50_MEAN
        ) / PIC50_STD

        # ----------------------------------------------------
        # Normalize predicted logBB
        # ----------------------------------------------------
        labels["normalized_logBB"] = (
            labels["predicted_logBB"].astype(float)
            - BBB_MEAN
        ) / BBB_STD

        # ----------------------------------------------------
        # Build canonical SMILES condition map
        # ----------------------------------------------------
        self.condition_map = {}

        invalid_label_smiles = 0

        for _, row in labels.iterrows():

            canonical = canonicalize_smiles(
                row["canonical_smiles"]
            )

            if canonical is None:
                invalid_label_smiles += 1
                continue

            self.condition_map[canonical] = (
                float(row["pIC50"]),
                float(row["normalized_pIC50"]),
                float(row["predicted_logBB"]),
                float(row["normalized_logBB"]),
            )

        # ----------------------------------------------------
        # Report label processing
        # ----------------------------------------------------
        print(
            f"Condition records loaded: "
            f"{len(labels):,}"
        )

        print(
            f"Canonical condition records: "
            f"{len(self.condition_map):,}"
        )

        print(
            f"Invalid condition SMILES: "
            f"{invalid_label_smiles:,}"
        )

        # ----------------------------------------------------
        # Retrieve every DrugGEN graph.
        #
        # Canonicalize the graph SMILES before matching.
        # ----------------------------------------------------
        graph_list = []

        missing_labels = 0
        invalid_graph_smiles = 0

        unmatched_smiles = []

        total_graphs = len(self)

        for i in range(total_graphs):

            data = self.get(i)

            smiles = getattr(
                data,
                "smiles",
                None,
            )

            if smiles is None:

                missing_labels += 1

                unmatched_smiles.append(
                    "<missing smiles>"
                )

                continue

            canonical = canonicalize_smiles(
                smiles
            )

            if canonical is None:

                invalid_graph_smiles += 1

                unmatched_smiles.append(
                    str(smiles)
                )

                continue

            values = self.condition_map.get(
                canonical
            )

            if values is None:

                missing_labels += 1

                unmatched_smiles.append(
                    str(smiles)
                )

                continue

            (
                pIC50,
                normalized_pIC50,
                predicted_logBB,
                normalized_logBB,
            ) = values

            # ------------------------------------------------
            # Three-dimensional condition
            #
            # [BACE1 target,
            #  normalized pIC50,
            #  normalized logBB]
            # ------------------------------------------------
            data.condition = torch.tensor(
                [
                    1.0,
                    normalized_pIC50,
                    normalized_logBB,
                ],
                dtype=torch.float32,
            )

            # ------------------------------------------------
            # Store original continuous values for verification
            # ------------------------------------------------
            data.pIC50 = torch.tensor(
                [pIC50],
                dtype=torch.float32,
            )

            data.predicted_logBB = torch.tensor(
                [predicted_logBB],
                dtype=torch.float32,
            )

            graph_list.append(
                data
            )

        # ----------------------------------------------------
        # Alignment verification
        # ----------------------------------------------------
        if missing_labels > 0:

            print()
            print("=" * 70)
            print("CONDITION ALIGNMENT ERROR")
            print("=" * 70)

            print(
                f"DrugGEN graphs:       {total_graphs:,}"
            )

            print(
                f"Matched graphs:       {len(graph_list):,}"
            )

            print(
                f"Missing conditions:   {missing_labels:,}"
            )

            print(
                f"Invalid graph SMILES: {invalid_graph_smiles:,}"
            )

            print()
            print(
                "First unmatched SMILES:"
            )

            for smiles in unmatched_smiles[:10]:
                print(
                    f"  {smiles}"
                )

            raise ValueError(
                f"Could not assign conditions to "
                f"{missing_labels} DrugGEN graphs."
            )

        # ----------------------------------------------------
        # Rebuild InMemoryDataset storage
        # ----------------------------------------------------
        self.data, self.slices = self.collate(
            graph_list
        )

        # ----------------------------------------------------
        # Final verification
        # ----------------------------------------------------
        print()
        print("-" * 70)
        print("CONDITION ALIGNMENT VERIFIED")
        print("-" * 70)

        print(
            f"DrugGEN graphs:       {total_graphs:,}"
        )

        print(
            f"Matched graphs:       {len(graph_list):,}"
        )

        print(
            "Missing conditions:   0"
        )

        print(
            "Condition dimension:  3"
        )

        print(
            "Condition definition:"
        )

        print(
            "  [BACE1=1, normalized pIC50, normalized predicted logBB]"
        )

        print()
        print(
            f"pIC50 normalization: "
            f"mean={PIC50_MEAN:.6f}, "
            f"std={PIC50_STD:.6f}"
        )

        print(
            f"logBB normalization: "
            f"mean={BBB_MEAN:.6f}, "
            f"std={BBB_STD:.6f}"
        )

