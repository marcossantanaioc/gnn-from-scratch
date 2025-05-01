import pytest
import torch
from rdkit import Chem
from collections.abc import Sequence
from typing import NamedTuple
import random
import dataclasses
from graphmodels import constants, data_utils


@dataclasses.dataclass(kw_only=True, frozen=True)
class SampleEntry:
    adj_matrix: torch.Tensor
    bond_features: torch.Tensor
    atom_features: torch.Tensor
    edge_indices: torch.Tensor
    target: torch.Tensor
    total_nodes: int
    total_edges: int


@dataclasses.dataclass(kw_only=True, frozen=True)
class SampleBatch:
    dsets: Sequence[SampleEntry]
    total_nodes: int
    total_edges: int


def _create_random_graph(
    min_num_nodes: int = 2,
    max_num_nodes: int = 50,
    n_atom_features: int = constants.NUM_ATOM_FEATURES,
    n_bond_features: int = constants.NUM_BOND_FEATURES,
):
    n_nodes = random.randint(min_num_nodes, max_num_nodes)

    target = torch.rand(1)

    adj_matrix = torch.rand(n_nodes, n_nodes)
    adj_matrix[adj_matrix > 0.5] = 1
    adj_matrix[adj_matrix <= 0.5] = 0

    # Make symmetric
    adj_matrix = adj_matrix + adj_matrix.T  # /2

    # Remove self-loops
    adj_matrix.fill_diagonal_(0)

    adj_matrix = torch.where(adj_matrix >= 1, 1, 0)

    edge_indices = torch.nonzero(adj_matrix).T

    atom_features = torch.rand(n_nodes, n_atom_features)
    bond_features = torch.rand(edge_indices.size(1), n_bond_features)

    return SampleEntry(
        adj_matrix=adj_matrix,
        bond_features=bond_features,
        atom_features=atom_features,
        edge_indices=edge_indices,
        target=target,
        total_nodes=n_nodes,
        total_edges=adj_matrix.sum(),
    )


def _generate_random_dataset(
    min_num_nodes: int = 2,
    max_num_nodes: int = 50,
    n_atom_features: int = constants.NUM_ATOM_FEATURES,
    n_bond_features: int = constants.NUM_BOND_FEATURES,
    num_examples: int = 10,
) -> Sequence[SampleEntry]:
    dsets = []
    total_nodes = 0
    total_edges = 0

    for _ in range(num_examples):
        dset = _create_random_graph()
        if dset.edge_indices.numel() > 0:
            total_nodes += dset.total_nodes
            total_edges += dset.total_edges
            dsets.append(dset)

    return SampleBatch(
        dsets=dsets, total_nodes=total_nodes, total_edges=total_edges
    )


class TestMPNNBatching:
    """
    Pytests
    """

    @pytest.fixture
    def batch(self) -> SampleBatch:
        return _generate_random_dataset(num_examples=64)

    def test_create_batch_edge_index(self, batch):
        collated_edge_indices = data_utils.create_batch_edge_index(
            [dset.edge_indices for dset in batch.dsets]
        )

        assert collated_edge_indices.size(1) == batch.total_edges

    def test_mpnn_collate_diag(self, batch):
        collated_dataset = data_utils.mpnn_collate_diag(batch.dsets)

        assert collated_dataset.atom_features.size(0) == batch.total_nodes


if __name__ == "__main__":
    pytest.main([__file__])
