import data_helpers
import pytest
import torch

from graphmodels import constants, data_utils
from graphmodels.datasets import mpnn_dataset


class TestMPNNBatching:
    """
    Pytests
    """

    @pytest.fixture
    def batch(self) -> list[mpnn_dataset.MPNNEntry]:
        dsets = data_helpers._generate_random_dataset(num_examples=64)
        return [
            mpnn_dataset.MPNNEntry(
                node_features=dset.node_features,
                edge_features=dset.edge_features,
                target=dset.target,
                adj_matrix=dset.adj_matrix,
                edge_indices=dset.edge_indices,
            )
            for dset in dsets.dsets
        ]

    # def test_create_batch_edge_index(self, batch):
    #     """Test batch edge indices."""
    #     collated_edge_indices = data_utils.create_batch_edge_index(
    #         [dset.edge_indices for dset in batch.dsets],
    #     )

    #     # Check number of edges match expected values for all elements.
    #     assert collated_edge_indices.size(1) == batch.total_edges

    #     # Check node index match the number of nodes
    #     assert collated_edge_indices.max().item() + 1 == batch.total_nodes

    def test_mpnn_collate_diag(self, batch):
        collated_dataset = data_utils.mpnn_collate_diag(batch)
        total_nodes = sum([b.node_features.shape[0] for b in batch])

        # Check features have the right shape
        assert collated_dataset.node_features.size(0) == total_nodes
        assert (
            collated_dataset.node_features.size(1)
            == constants.NUM_NODE_FEATURES
        )
        assert (
            collated_dataset.edge_features.size(1)
            == constants.NUM_EDGE_FEATURES
        )

        # Check batch vector is correct
        assert collated_dataset.batch_vector.size(0) == total_nodes
        assert torch.unique(collated_dataset.batch_vector).size(0) == len(
            batch,
        )


if __name__ == "__main__":
    pytest.main([__file__])
