import pytest
import torch

from graphmodels.models import ngf_model


class TestNGFModel:
    """Pytests"""

    @pytest.mark.parametrize(
        "input_features, h_units, output_units, radius, batch_size, num_atoms",
        [
            (10, 32, 5, 3, 2, 5),
            (20, 64, 10, 4, 3, 7),
            (5, 16, 3, 2, 1, 3),
        ],
    )
    def test_neural_graph_fingerprint_model_output_shape(
        self,
        input_features,
        h_units,
        output_units,
        radius,
        batch_size,
        num_atoms,
    ):
        # Create an instance of the NeuralGraphFingerprintModel
        model = ngf_model.NeuralGraphFingerprintModel(
            input_features,
            h_units,
            output_units,
            radius,
        )

        # Define dummy input data: atom features and adjacency matrix
        atom_feats = torch.randn(batch_size, num_atoms, input_features)
        # Adjacency matrix: (batch_size, num_atoms, num_atoms).
        adj_matrix = torch.zeros(batch_size, num_atoms, num_atoms)
        for i in range(num_atoms):
            adj_matrix[:, i, (i + 1) % num_atoms] = 1
            adj_matrix[:, i, (i - 1 + num_atoms) % num_atoms] = 1

        input_data = (atom_feats, adj_matrix)

        output = model(input_data)

        expected_shape = torch.Size([batch_size, output_units])
        assert output.shape == expected_shape, (
            f"Output shape was {output.shape}, expected {expected_shape}.  "
            f"Input shape was {input_data[0].shape}, {input_data[1].shape}. "
            f"Model parameters: input_features={input_features}, "
            f"h_units={h_units}, output_units={output_units}, "
            f"radius={radius}, batch_size={batch_size}, num_atoms={num_atoms}"
        )
        print("Test passed: Output shape is correct!")


if __name__ == "__main__":
    pytest.main([__file__])
