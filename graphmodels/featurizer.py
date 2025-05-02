import torch
import torch.nn.functional as F  # noqa: N812
from typing import NamedTuple
from rdkit import Chem

from graphmodels import constants


class NoAtomError(Exception):
    pass


class GraphConnectivity(NamedTuple):
    """Stores connectivity metrics."""

    adj_matrix: torch.Tensor
    edge_index: torch.Tensor


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
        torch.tensor(constants.EDGE_TYPES[bond.GetBondType()]),
        num_classes=constants.MAX_EDGE_TYPES,
    )

    is_in_ring = torch.tensor(int(bond.IsInRing())).unsqueeze(-1)

    return torch.cat([bond_type, is_conjugated, is_in_ring], dim=-1)


def featurize_bonds_per_atom(molecule: Chem.Mol) -> torch.Tensor:
    """Generates bond features represented as an atom-to-atom adjacency tensor.

    This function creates a tensor of shape (N, N, F), where:

    - N: is the number of atoms (nodes) in the molecule.
    - F: is the dimensionality of the bond features (i.e., 24).

    The value at index (i, j, :) in the tensor represents the features
    of the bond between atom i and atom j. If no bond exists between
    these atoms, the feature vector is all zeros. The representation is
    symmetric, meaning the features for the bond between atom i and j are
    the same as those between atom j and i.

    Args:
        molecule: An RDKit Chem.Mol object.

    Returns:
        A tensor of shape (N, N, F) containing the calculated bond features.
    """

    if molecule.GetNumBonds() == 0:
        return torch.zeros(
            (
                molecule.GetNumAtoms(),
                molecule.GetNumAtoms(),
                constants.NUM_EDGE_FEATURES,
            ),
        )
    all_bond_features = torch.zeros(
        (
            molecule.GetNumAtoms(),
            molecule.GetNumAtoms(),
            constants.NUM_EDGE_FEATURES,
        ),
    )
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_features = _featurize_one_bond(bond)

        all_bond_features[i, j] = bond_features
        all_bond_features[j, i] = bond_features
    return all_bond_features.to(torch.float32)


def get_graph_connectivity(molecule: Chem.Mol) -> torch.Tensor:
    """Generates adjacency matrix and edge indices for a molecular graph.

    This function constructs atom-level connectivity information from an RDKit molecule.
    It returns:

    - An adjacency matrix of shape (N, N), where N is the number of atoms. Entry (i, j)
      is 1 if a bond exists between atom i and atom j, and 0 otherwise.

    - An edge index tensor of shape (2 * num_bonds, 2), listing all bonded atom pairs
      in both directions (i -> j and j -> i), suitable for use in graph neural networks.

    Args:
        molecule: An RDKit `Chem.Mol` object representing the molecule.

    Returns:
        A `GraphConnectivity` object containing:
            - adj_matrix: A (N, N) tensor representing the adjacency matrix.
            - edge_index: A (2, num_bonds*2) tensor of directed edge indices.
    tensor of shape (N, N, F) containing the calculated bond features.
    """
    
    num_atoms = molecule.GetNumAtoms()
    adj_matrix = torch.zeros(num_atoms, num_atoms)
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1

    return GraphConnectivity(
        adj_matrix=adj_matrix.long(), edge_index=torch.nonzero(torch.triu(adj_matrix)).T.contiguous().long()
    )


def featurize_bonds(molecule: Chem.Mol) -> torch.Tensor:
    """Generates a compact tensor of bond features for a given molecule.

    This function iterates through each bond in the input `molecule` and
    calculates a set of features for it. The resulting tensor has a shape of
    (B, F), where:

    - B: represents the total number of bonds in the molecule.
    - F: is the dimensionality of the bond features (currently 24).

    This function provides a more memory-efficient representation of bond
    features compared to `featurize_bonds_per_atom`. While
    `featurize_bonds_per_atom` creates an (N, N, F) tensor (where N is the
    number of atoms), this function directly stores the features for each
    existing bond in a (B, F) tensor.
    """
    if molecule.GetNumBonds() == 0:
        return torch.zeros(1, constants.NUM_EDGE_FEATURES).to(torch.float32)

    bond_features = []
    for bond in molecule.GetBonds():
        feats = _featurize_one_bond(bond)
        bond_features.append(feats)
    return torch.stack(bond_features).to(torch.float32)


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

    if molecule.GetNumAtoms() == 0:
        raise NoAtomError("Cannot featurize a molecule with no atoms.")

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

    return torch.stack(raw_features).to(torch.float32)