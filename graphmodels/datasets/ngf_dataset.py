import torch
from rdkit import Chem
from torch.utils import data as torch_data
import torch.nn.functional as F
import dataclasses
from graphmodels import featurizer


@dataclasses.dataclass(kw_only=True, frozen=True)
class NeuralFingerprintEntry:
    """Class to store input data for neuralgraph fingerprint."""

    node_features: torch.Tensor
    edge_features: torch.Tensor
    adj_matrix: torch.Tensor
    target: torch.Tensor


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

        node_features = featurizer.featurize_atoms(mol).to(torch.float32)
        edge_features = featurizer.featurize_bonds_per_atom(mol).to(
            torch.float32
        )
        adj_matrix = torch.tensor(Chem.GetAdjacencyMatrix(mol)).to(
            torch.float32,
        )

        return node_features, edge_features, adj_matrix

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
        node_features, edge_features, adj_matrix = self.transform(
            self.smiles[idx],
        )
        target = torch.tensor(self.targets[idx])

        return NeuralFingerprintEntry(
            node_features=node_features,
            edge_features=edge_features,
            target=target,
            adj_matrix=adj_matrix,
        )
