import torch
import torch.nn.functional as F  # noqa: N812
from rdkit import Chem

from neuralfingerprint import constants


def _featurize_one_bond(bond: Chem.Bond) -> torch.Tensor:
    """Generates a tensor of bond features.
    This function returns a tensor representing bond features for every
    bond in the input molecule. Bond features consists of bond type,
    conjugation and information whether its on a ring.

    This implementation differs from Duvenaud et al, in that it one-hot encode
    bond types.
    For the original implementation, see:
    https://github.com/HIPS/neural-fingerprint/blob/master/neuralfingerprint/features.py

    Args:
        molecule: a Chem.Mol object.

    Returns:
        A tensor of bond features of shape (N, 24), where N is the number of
        bonds in the molecule.


    """
    is_conjugated = torch.tensor(int(bond.GetIsConjugated())).unsqueeze(-1)
    bond_type = F.one_hot(
        torch.tensor(constants.BOND_TYPES[bond.GetBondType()]),
        num_classes=constants.MAX_BOND_TYPES,
    )

    is_in_ring = torch.tensor(int(bond.IsInRing())).unsqueeze(-1)

    return torch.cat([bond_type, is_conjugated, is_in_ring], dim=-1)


def featurize_bonds(molecule: Chem.Mol) -> torch.Tensor:
    all_bond_features = torch.zeros(
        (
            molecule.GetNumAtoms(),
            molecule.GetNumAtoms(),
            constants.NUM_BOND_FEATURES,
        ),
    )
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_features = _featurize_one_bond(bond)

        all_bond_features[i, j] = bond_features
        all_bond_features[j, i] = bond_features
    return all_bond_features


def featurize_atoms(molecule: Chem.Mol) -> torch.Tensor:
    """Generates a tensor of atom features.

    Atom features consist of:
    - Atomic number. We expanded to the whole periodic table.
    - Degree (number of directly bonded atoms)
    - Implicit valence
    - Total number of hydrogen atoms
    - Aromaticity flag

    For the original implementation, see:
    https://github.com/HIPS/neural-fingerprint/blob/master/neuralfingerprint/features.py
    Args:
        molecule: a Chem.Mol object.

    Returns:
        A tensor of atom features of shape (N, 135), where N is the number of
        bonds in the molecule.


    """
    atoms = molecule.GetAtoms()
    raw_features = []
    for atom in atoms:
        atomic_number_one_hot = F.one_hot(
            torch.tensor(atom.GetAtomicNum()),
            num_classes=constants.MAX_ATOMIC_NUMBER,
        )
        degree_one_hot = F.one_hot(
            torch.tensor(atom.GetDegree()),
            num_classes=constants.MAX_DEGREE,
        )
        valence_one_hot = F.one_hot(
            torch.tensor(atom.GetImplicitValence()),
            num_classes=constants.MAX_VALENCE,
        )
        num_hydrogens_one_hot = F.one_hot(
            torch.tensor(atom.GetTotalNumHs()),
            num_classes=constants.MAX_HYDROGEN_ATOMS,
        )
        is_aromatic = torch.tensor(atom.GetIsAromatic()).unsqueeze(-1)

        atom_feat = torch.cat(
            [
                atomic_number_one_hot,
                degree_one_hot,
                valence_one_hot,
                num_hydrogens_one_hot,
                is_aromatic,
            ],
            -1,
        )
        raw_features.append(atom_feat)

    return torch.stack(raw_features)
