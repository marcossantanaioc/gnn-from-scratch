import dataclasses

import torch
import torch.nn.functional as F  # noqa: N812
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from rdkit import Chem
from torch.utils import data as torch_data
from typeguard import typechecked as typechecker

from graphmodels import featurizer


@jt(typechecker=typechecker)
@dataclasses.dataclass(kw_only=True, frozen=True)
class MPNNEntry:
    """Class to store input data for MPNN v1."""

    node_features: Float[torch.Tensor, "nodes node_features"]
    edge_features: Float[torch.Tensor, "edges edge_features"]
    adj_matrix: Int[torch.Tensor, "nodes nodes"]
    edge_indices: Int[torch.Tensor, "2 edges"]
    target: Float[torch.Tensor, ""]


@jt(typechecker=typechecker)
class MPNNDataset(torch_data.Dataset):
    """Creates a molecule dataset based on the MPNN v1 model.

    This dataset takes lists of SMILES strings and corresponding target values
    and prepares them for use with a Message Passing Neural Network (MPNN).
    It featurizes molecules into atom and bond features and constructs
    adjacency matrices, optionally adding a master node. The dataset returns
    `MPNNEntry` objects containing these features, the target, the adjacency
    matrix, and the edge indices.

    Args:
        smiles: A tuple of SMILES strings representing the molecules.
        targets: A tuple of float values corresponding to the target property
            for each molecule in `smiles`.
        add_master_node: Whether to include a master node connected to all
        nodes.
    """

    def __init__(
        self,
        smiles: tuple[str, ...],
        targets: tuple[float, ...],
        add_master_node: bool = False,
    ):
        self.smiles = smiles
        self.targets = targets
        self.add_master_node = add_master_node

    def __len__(self):
        return len(self.smiles)

    def transform(
        self,
        smiles: str,
    ) -> tuple[
        Float[torch.Tensor, "nodes node_features"],
        Float[torch.Tensor, "edges edge_features"],
        Int[torch.Tensor, "nodes nodes"],
        Int[torch.Tensor, "2 edges"],
    ]:
        """Featurize one molecule.

        Args:
            smiles: The input SMILES to be featurized
        Returns:
            A tuple with atom and bond features, and the prediction target.
        """
        mol = Chem.MolFromSmiles(smiles)

        node_features = featurizer.featurize_atoms(mol)

        edge_features = featurizer.featurize_bonds(mol)

        adj_matrix, edge_index = featurizer.get_graph_connectivity(
            mol,
        )

        if self.add_master_node:
            adj_matrix = F.pad(adj_matrix, (0, 1, 0, 1), value=1)
            adj_matrix[-1, -1] = 0  # Force no self connection in master node.
            node_features = F.pad(node_features, (0, 0, 0, 1), value=0)

        return node_features, edge_features, adj_matrix, edge_index

    def __getitem__(
        self,
        idx,
    ) -> MPNNEntry:
        """Returns one element from the dataset.

        Args:
            idx: index to retrieve.

        Returns:
            A tuple with atom and bond features, and the prediction target.


        """
        node_features, edge_features, adj_matrix, edge_indices = (
            self.transform(
                self.smiles[idx],
            )
        )

        target = torch.tensor(self.targets[idx])

        return MPNNEntry(
            node_features=node_features,
            edge_features=edge_features,
            target=target,
            adj_matrix=adj_matrix,
            edge_indices=edge_indices,
        )
