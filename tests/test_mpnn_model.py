import data_helpers
import pytest
import torch

from graphmodels import data_utils
from graphmodels.datasets import mpnn_dataset
from graphmodels.models import mpnn


class TestMPNNModel:
    """Pytests"""

    @pytest.mark.parametrize(
        "bs,atoms,n_feats,e_feats,out_units,towers,h_features,n_updates,n_readout",
        [
            (
                2,
                29,
                64,
                24,
                1,
                8,
                200,
                3,
                3,
            ),
            (3, 20, 64, 32, 2, 8, 512, 2, 2),
            (5, 15, 136, 16, 5, 8, 100, 1, 1),
        ],  # Add multiple test cases
    )
    def test_mpnnv1_model_output_shape(
        self,
        bs,
        atoms,
        n_feats,
        e_feats,
        h_features,
        towers,
        n_updates,
        n_readout,
        out_units,
    ):
        dset_entries = [
            mpnn_dataset.MPNNEntry(
                node_features=dset.node_features,
                edge_features=dset.edge_features,
                target=dset.target,
                adj_matrix=dset.adj_matrix,
                edge_indices=dset.edge_indices,
            )
            for dset in data_helpers._generate_random_dataset(
                num_examples=bs,
                min_num_nodes=atoms,
                max_num_nodes=atoms * 5,
                n_edge_features=e_feats,
                n_node_features=n_feats,
            ).dsets
        ]
        batch = data_utils.mpnn_collate_diag(dset_entries)

        # Create an instance of the MPNN
        model = mpnn.MPNNv1(
            n_node_features=n_feats,
            n_edge_features=e_feats,
            n_hidden_features=h_features,
            n_update_steps=n_updates,
            n_readout_steps=n_readout,
            n_out_features=out_units,
            n_towers=towers,
        )

        # Pass the input data through the model
        output = model(
            node_features=batch.node_features,
            edge_features=batch.edge_features,
            edge_index=batch.edge_index,
            batch_vector=batch.batch_vector,
        )
        expected_shape = torch.Size([bs, out_units])
        assert output.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__])
