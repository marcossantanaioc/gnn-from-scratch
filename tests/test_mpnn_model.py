import pytest
import torch

from graphmodels.models import mpnn
from graphmodels import data_utils
import data_helpers


class TestMPNNModel:
    """Pytests"""

    @pytest.mark.parametrize(
        "num_atoms, num_bonds, n_node_features, n_edge_features, n_out_features, n_towers, n_hidden_features, n_update_steps, n_readout_steps",
        [
            (
                29,
                32,
                64,
                24,
                1,
                8,
                200,
                3,
                3,
            ),
            (20, 26, 64, 32, 2, 8, 512, 2, 2),
            (15, 18, 136, 16, 5, 8, 100, 1, 1),
        ],  # Add multiple test cases
    )
    def test_mpnnv1_model_output_shape(
        self,
        num_atoms,
        num_bonds,
        n_node_features,
        n_edge_features,
        n_hidden_features,
        n_towers,
        n_update_steps,
        n_readout_steps,
        n_out_features,
    ):
        # Create an instance of the MPNN
        model = mpnn.MPNNv1(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_hidden_features=n_hidden_features,
            n_update_steps=n_update_steps,
            n_readout_steps=n_readout_steps,
            n_out_features=n_out_features,
            n_towers=n_towers,
        )

        batch = data_utils.mpnn_collate_diag(
            data_helpers._generate_random_dataset(
                min_num_nodes=num_atoms,
                max_num_nodes=num_atoms,
                n_node_features=n_node_features,
                n_edge_features=n_edge_features,
                num_examples=64,
            ).dsets
        )

        # Ensure the adjacency matrix is used correctly.
        input_data = (
            batch.node_features,
            batch.edge_features,
            batch.edge_index,
            batch.batch_vector,
        )

        # Pass the input data through the model
        output = model(input_data)

        # Assert that the output shape is as expected: (batch_size, n_output_units)
        expected_shape = torch.Size([64, n_out_features])
        assert output.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__])
