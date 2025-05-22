import dataclasses

import torch
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from rdkit import Chem
from torch.utils import data as torch_data
from typeguard import typechecked as typechecker

from graphmodels import featurizer


@jt(typechecker=typechecker)
@dataclasses.dataclass(kw_only=True, frozen=True)
class NeuralFingerprintEntry:
    """Class to store input data for neuralgraph fingerprint."""

    node_features: Float[torch.Tensor, "nodes node_features"]
    edge_features: Float[torch.Tensor, "nodes nodes edge_features"]
    adj_matrix: Int[torch.Tensor, "nodes nodes"]
    target: Float[torch.Tensor, ""]


@jt(typechecker=typechecker)
class NeuralFingerprintDataset(torch_data.Dataset):
    """Creates a molecule dataset based on neural fingerprints."""

    def __init__(self, smiles: tuple[str, ...], targets: tuple[float, ...]):
        self.smiles = smiles
        self.targets = targets

    def __len__(self) -> int:
        return len(self.smiles)

    def transform(
        self,
        smiles: str,
    ) -> tuple[
        Float[torch.Tensor, "nodes node_features"],
        Float[torch.Tensor, "edges edge_features"],
        Int[torch.Tensor, "nodes nodes"],
    ]:
        """Featurize one molecule.

        Args:
            smiles: The input SMILES to be featurized
        Returns:
            A tuple with atom and bond features, and the prediction target.
        """
        mol = Chem.MolFromSmiles(smiles)

        node_features = featurizer.featurize_atoms(mol).to(torch.float32)
        edge_features = featurizer.featurize_bonds_per_atom(mol).to(
            torch.float32,
        )
        adj_matrix = torch.tensor(Chem.GetAdjacencyMatrix(mol)).to(
            torch.long,
        )

        return node_features, edge_features, adj_matrix

    def __getitem__(
        self,
        idx,
    ) -> NeuralFingerprintEntry:
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
