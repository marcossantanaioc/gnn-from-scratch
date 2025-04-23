import torch
from rdkit import Chem
from torch.utils import data as torch_data
import dataclasses
from graphmodels import featurizer


@dataclasses.dataclass(kw_only=True, frozen=True)
class NeuralFingerprintEntry:
    """Class to store input data for neuralgraph fingerprint."""

    atom_features: torch.Tensor
    bond_features: torch.Tensor
    adj_matrix: torch.Tensor
    target: torch.Tensor


@dataclasses.dataclass(kw_only=True, frozen=True)
class MPNNEntry:
    """Class to store input data for MPNN v1."""

    atom_features: torch.Tensor
    bond_features: torch.Tensor
    adj_matrix: torch.Tensor
    edge_indices: torch.Tensor
    target: torch.Tensor


class MPNNDataset(torch_data.Dataset):
    """Creates a molecule dataset based on MPNN v1 model."""

    def __init__(self, smiles: tuple[str, ...], targets: tuple[float, ...]):
        self.smiles = smiles
        self.targets = targets

    def __len__(self):
        return len(self.smiles)

    def transform(self, smiles: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Featurize one molecule.

        Args:
            smiles: The input SMILES to be featurized
        Returns:
            A tuple with atom and bond features, and the prediction target.
        """
        mol = Chem.MolFromSmiles(smiles)

        atom_features = featurizer.featurize_atoms(mol)

        bond_features = featurizer.featurize_bonds(mol)

        adj_matrix = torch.tril(
            torch.tensor(Chem.GetAdjacencyMatrix(mol)).to(torch.float32)
        )

        return atom_features, bond_features, adj_matrix

    def __getitem__(
        self,
        idx,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns one element from the dataset.

        Args:
            idx: index to retrieve.

        Returns:
            A tuple with atom and bond features, and the prediction target.


        """
        atom_features, bond_features, adj_matrix = self.transform(
            self.smiles[idx],
        )
        target = torch.tensor(self.targets[idx])
        edge_indices = torch.nonzero(adj_matrix)[:, [1, 0]]
        return MPNNEntry(
            atom_features=atom_features,
            bond_features=bond_features,
            target=target,
            adj_matrix=adj_matrix,
            edge_indices=edge_indices,
        )


class NeuralFingerprintDataset(torch_data.Dataset):
    """Creates a molecule dataset based on neural fingerprints."""

    def __init__(self, smiles: tuple[str, ...], targets: tuple[float, ...]):
        self.smiles = smiles
        self.targets = targets

    def __len__(self):
        return len(self.smiles)

    def transform(self, smiles: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Featurize one molecule.

        Args:
            smiles: The input SMILES to be featurized
        Returns:
            A tuple with atom and bond features, and the prediction target.
        """
        mol = Chem.MolFromSmiles(smiles)

        atom_features = featurizer.featurize_atoms(mol).to(torch.float32)
        bond_features = featurizer.featurize_bonds_per_atom(mol).to(
            torch.float32
        )
        adj_matrix = torch.tensor(Chem.GetAdjacencyMatrix(mol)).to(
            torch.float32,
        )

        return atom_features, bond_features, adj_matrix

    def __getitem__(
        self,
        idx,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns one element from the dataset.

        Args:
            idx: index to retrieve.

        Returns:
            A tuple with atom and bond features, and the prediction target.


        """
        atom_features, bond_features, adj_matrix = self.transform(
            self.smiles[idx],
        )
        target = torch.tensor(self.targets[idx])
        return NeuralFingerprintEntry(
            atom_features=atom_features,
            bond_features=bond_features,
            target=target,
            adj_matrix=adj_matrix,
        )
