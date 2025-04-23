import pytest
import torch

from graphmodels import models


@pytest.mark.parametrize(
    "n_input_features, n_hidden_units, n_output_units, radius, batch_size, num_atoms",
    [
        (10, 32, 5, 3, 2, 5),
        (20, 64, 10, 4, 3, 7),
        (5, 16, 3, 2, 1, 3),
    ],  # Add multiple test cases
)
def test_neural_graph_fingerprint_model_output_shape(
    n_input_features,
    n_hidden_units,
    n_output_units,
    radius,
    batch_size,
    num_atoms,
):
    """
    Test case to verify the output shape of the NeuralGraphFingerprintModel.
    It checks if the model produces the expected output shape given a sample
    input of atom features and adjacency matrix.  This version uses pytest.mark.parametrize
    to run the test with multiple sets of parameters.  All hardcoded variables are now included
    in the parameterization.
    """
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


if __name__ == "__main__":
    pytest.main([__file__])
