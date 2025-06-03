import data_helpers
import pytest
import torch

from graphmodels import data_utils
from graphmodels.datasets import mpnn_dataset
from graphmodels.models import gat


class TestGATModels:
    """Pytests"""

    @pytest.mark.parametrize(
        "bs,atoms,n_feats,h_features,out_units,heads,num_layers,output_level",
        [
            (
                2,
                29,
                64,
                136,
                1,
                2,
                1,
                "graph",
            ),
            (
                3,
                20,
                64,
                200,
                4,
                2,
                3,
                "node",
            ),
            (
                5,
                15,
                136,
                64,
                2,
                2,
                2,
                "node",
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
        num_layers,
        heads,
        output_level,
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
            num_layers=num_layers,
            num_heads=heads,
            scaling=0.2,
            dropout=0.5,
            output_level=output_level,
        )

        # Pass the input data through the model
        output = model(
            node_features=batch.node_features,
            edge_index=batch.edge_index,
            batch_vector=batch.batch_vector,
        )
        if output_level == "graph":
            expected_shape = torch.Size([bs, out_units])
            assert output.shape == expected_shape
        elif output_level == "node":
            assert output.shape == torch.Size(
                [len(batch.node_features), out_units],
            )


if __name__ == "__main__":
    pytest.main([__file__])
