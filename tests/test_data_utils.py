import pytest
import torch
from rdkit import Chem
from collections.abc import Sequence
from typing import NamedTuple
import random
import dataclasses
from graphmodels import constants, data_utils
import data_helpers


class TestMPNNBatching:
    """
    Pytests
    """

    @pytest.fixture
    def batch(self) -> data_helpers.SampleBatch:
        return data_helpers._generate_random_dataset(num_examples=64)

    def test_create_batch_edge_index(self, batch):
        """Test batch edge indices."""
        collated_edge_indices = data_utils.create_batch_edge_index(
            [dset.edge_indices for dset in batch.dsets]
        )

        # Check number of edges match expected values for all elements.
        assert collated_edge_indices.size(1) == batch.total_edges

        # Check node index match the number of nodes
        assert collated_edge_indices.max().item() + 1 == batch.total_nodes

    def test_mpnn_collate_diag(self, batch):
        collated_dataset = data_utils.mpnn_collate_diag(batch.dsets)

        # Check features have the right shape
        assert collated_dataset.node_features.size(0) == batch.total_nodes
        assert (
            collated_dataset.node_features.size(1)
            == constants.NUM_ATOM_FEATURES
        )
        assert (
            collated_dataset.edge_features.size(1)
            == constants.NUM_BOND_FEATURES
        )

        # Check batch vector is correct
        assert collated_dataset.batch_vector.size(0) == batch.total_nodes
        assert torch.unique(collated_dataset.batch_vector).size(0) == len(
            batch.dsets
        )


if __name__ == "__main__":
    pytest.main([__file__])
