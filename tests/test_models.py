import pytest
import torch

from graphmodels import models, data_utils
import data_helpers


class TestModels:
    """Pytests"""

    @pytest.mark.parametrize(
        "n_input_features, n_hidden_units, n_output_units, radius, batch_size, num_atoms",
        [
            (10, 32, 5, 3, 2, 5),
            (20, 64, 10, 4, 3, 7),
            (5, 16, 3, 2, 1, 3),
        ],  # Add multiple test cases
    )
    def test_neural_graph_fingerprint_model_output_shape(
        self,
        n_input_features,
        n_hidden_units,
        n_output_units,
        radius,
        batch_size,
        num_atoms,
    ):
        # Create an instance of the NeuralGraphFingerprintModel
        model = models.NeuralGraphFingerprintModel(
            n_input_features, n_hidden_units, n_output_units, radius
        )

        # Define dummy input data: atom features and adjacency matrix
        atom_feats = torch.randn(batch_size, num_atoms, n_input_features)
        # Adjacency matrix: (batch_size, num_atoms, num_atoms).
        adj_matrix = torch.zeros(batch_size, num_atoms, num_atoms)
        for i in range(num_atoms):
            adj_matrix[:, i, (i + 1) % num_atoms] = 1
            adj_matrix[:, i, (i - 1 + num_atoms) % num_atoms] = 1

        # Ensure the adjacency matrix is used correctly.
        input_data = (atom_feats, adj_matrix)

        # Pass the input data through the model
        output = model(input_data)

        # Assert that the output shape is as expected: (batch_size, n_output_units)
        expected_shape = torch.Size([batch_size, n_output_units])
        assert output.shape == expected_shape, (
            f"Output shape was {output.shape}, expected {expected_shape}.  "
            f"Input shape was {input_data[0].shape}, {input_data[1].shape}. "
            f"Model parameters: n_input_features={n_input_features}, "
            f"n_hidden_units={n_hidden_units}, n_output_units={n_output_units}, "
            f"radius={radius}, batch_size={batch_size}, num_atoms={num_atoms}"
        )
        print("Test passed: Output shape is correct!")

    @pytest.mark.parametrize(
        "num_atoms, num_bonds, n_node_features, n_edge_features, n_out_features, n_bond_hidden_features, n_hidden_features, n_message_passes, n_update_layers, n_readout_steps",
        [
            (
                29,
                32,
                100,
                24,
                1,
                200,
                200,
                3,
                3,
                3,
            ),
            (20, 26, 50, 32, 2, 512, 512, 2, 2, 2),
            (15, 18, 136, 16, 5, 100, 100, 1, 1, 1),
        ],  # Add multiple test cases
    )
    def test_mpnnv1_model_output_shape(
        self,
        num_atoms,
        num_bonds,
        n_node_features,
        n_edge_features,
        n_bond_hidden_features,
        n_hidden_features,
        n_message_passes,
        n_update_layers,
        n_readout_steps,
        n_out_features,
    ):
        # Create an instance of the NeuralGraphFingerprintModel
        model = models.MPNNv1(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_bond_hidden_features=n_bond_hidden_features,
            n_hidden_features=n_hidden_features,
            n_message_passes=n_message_passes,
            n_update_layers=n_update_layers,
            n_readout_steps=n_readout_steps,
            n_out_features=n_out_features,
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
            batch.edge_features,
            batch.node_features,
            batch.edge_index,
            batch.batch_vector,
        )

        # Pass the input data through the model
        output = model(input_data)

        # Assert that the output shape is as expected: (batch_size, n_output_units)
        expected_shape = torch.Size([64, n_out_features])
        assert output.shape == expected_shape


#     min_num_nodes: int = 2,
#     max_num_nodes: int = 50,
#     n_node_features: int = constants.NUM_ATOM_FEATURES,
#     n_edge_features: int = constants.NUM_BOND_FEATURES,
#     num_examples: int = 10,

if __name__ == "__main__":
    pytest.main([__file__])
