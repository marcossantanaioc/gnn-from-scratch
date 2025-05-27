import data_helpers
import pytest
import torch

from graphmodels import data_utils
from graphmodels.datasets import mpnn_dataset
from graphmodels.models import gat


class TestGATModels:
    """Pytests"""

    def test_gat_model_raises_bad_output(self):
        with pytest.raises(ValueError):
            gat.GATModel(
                n_hidden_features=10,
                n_node_features=10,
                n_out_channels=1,
                n_layers=2,
                scaling=0.2,
                dropout=0.5,
                agg_method="concat",
                output_level="FALSE",
            )

    def test_gat_model_raises_bad_agg_method(self):
        with pytest.raises(ValueError):
            gat.GATModel(
                n_hidden_features=10,
                n_node_features=10,
                n_out_channels=1,
                n_layers=2,
                scaling=0.2,
                dropout=0.5,
                agg_method="FALSE",
                output_level="graph",
            )

    @pytest.mark.parametrize(
        "bs,atoms,n_feats,h_features,out_units,n_layers",
        [
            (
                2,
                29,
                64,
                200,
                1,
                1,
            ),
            (
                3,
                20,
                64,
                512,
                2,
                3,
            ),
            (
                5,
                15,
                136,
                100,
                3,
                5,
            ),
        ],  # Add multiple test cases
    )
    def test_gat_model_output_shape(
        self,
        bs,
        atoms,
        n_feats,
        h_features,
        out_units,
        n_layers,
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
                n_edge_features=10,
                n_node_features=n_feats,
            ).dsets
        ]
        batch = data_utils.mpnn_collate_diag(dset_entries)

        # Create an instance of the GAT
        model = gat.GATModel(
            n_hidden_features=h_features,
            n_node_features=n_feats,
            n_out_channels=out_units,
            n_layers=n_layers,
            scaling=0.2,
            dropout=0.5,
            agg_method="concat",
            output_level="graph",
        )

        # Pass the input data through the model
        output = model(
            node_features=batch.node_features,
            edge_index=batch.edge_index,
            batch_vector=batch.batch_vector,
        )
        expected_shape = torch.Size([bs, out_units])
        assert output.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__])
