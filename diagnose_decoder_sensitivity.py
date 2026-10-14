def decode_graph_to_smiles(
    node_graph,
    edge_graph,
    atom_decoder,
    bond_decoder,
):
    """
    Decode a generated graph into canonical SMILES.

    Atom class 0 (atomic number 0) is padding.
    BondType.ZERO represents the absence of a bond.
    """

    try:
        # Convert graph predictions into discrete classes.
        node_classes = (
            torch.argmax(node_graph, dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        edge_classes = (
            torch.argmax(edge_graph, dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        mol = Chem.RWMol()
        atom_indices = {}

        # --------------------------------------------------
        # ADD REAL ATOMS; IGNORE PADDING
        # --------------------------------------------------
        for i, atom_class in enumerate(node_classes):
            atomic_number = atom_decoder.get(int(atom_class))

            # Atomic number 0 is the padding class.
            if atomic_number is None or int(atomic_number) == 0:
                continue

            atom = Chem.Atom(int(atomic_number))
            atom_indices[i] = mol.AddAtom(atom)

        # There is no molecule if every position is padding.
        if mol.GetNumAtoms() == 0:
            return None

        # --------------------------------------------------
        # ADD REAL BONDS; IGNORE ZERO-ORDER BONDS
        # --------------------------------------------------
        for i in range(len(edge_classes)):
            for j in range(i + 1, len(edge_classes)):
                # Both endpoints must be real atoms.
                if i not in atom_indices or j not in atom_indices:
                    continue

                bond_class = int(edge_classes[i, j])
                bond_type = bond_decoder.get(bond_class)

                # Ignore missing and padding/no-bond classes.
                if (
                    bond_type is None
                    or bond_type == Chem.rdchem.BondType.ZERO
                ):
                    continue

                mol.AddBond(
                    atom_indices[i],
                    atom_indices[j],
                    bond_type,
                )

        # --------------------------------------------------
        # VALIDATE AND GENERATE CANONICAL SMILES
        # --------------------------------------------------
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)

        smiles = Chem.MolToSmiles(
            mol,
            canonical=True,
        )

        return smiles if smiles else None

    except Exception:
        # Invalid generated graphs are rejected.
        return None