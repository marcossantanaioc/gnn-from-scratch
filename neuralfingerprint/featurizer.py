from rdkit import Chem
import torch


def featurize_bonds(molecule: Chem.Mol) -> torch.Tensor:
    """Generates a tensor of bond features.
    This function returns a tensor representing bond features for every
    bond in the input molecule. Bond features consists of bond type,
    conjugation and information whether its on a ring.

    We use `bond.GetBondTypeAsDouble()` to get a numeric value for each type:

    Single: 1.0
    Aromatic: 1.5
    Double: 2.0
    Triple: 3.0

    Args:
        molecule: a Chem.Mol object.

    Returns:
        A tensor of bond features of shape (N, 3), where N is the number of
        bonds in the molecule.


    """
    bonds = molecule.GetBonds()
    raw_features = []
    for bond in bonds:
        is_conjugated = bond.GetIsConjugated()
        bond_type = int(bond.GetBondTypeAsDouble())
        is_in_ring = int(bond.IsInRing())
        raw_features.append((bond_type, is_conjugated, is_in_ring))

    return torch.as_tensor(raw_features)


def featurize_atoms(molecule: Chem.Mol) -> torch.Tensor:
    """Generates a tensor of atom features.
    Atom features consists of atomic number, degree (num. bond atoms), valence,
    total number of hydrogen atoms and an aromaticity flag.

    Args:
        molecule: a Chem.Mol object.

    Returns:
        A tensor of atom features of shape (N, 5), where N is the number of
        bonds in the molecule.


    """
    atoms = molecule.GetAtoms()
    features = []
    for atom in atoms:
        atomic_number = atom.GetAtomicNum()
        degree = atom.GetDegree()
        valence = atom.GetImplicitValence()
        num_hydrogens = atom.GetTotalNumHs()
        is_aromatic = atom.GetIsAromatic()
        features.append([atomic_number, degree, valence,
                        num_hydrogens, is_aromatic])
    return torch.as_tensor(features)
