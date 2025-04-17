import pytest
import torch

from graphmodels import models

@pytest.mark.parametrize(
    "batch_size, n_atoms, atom_feat_dim, bond_feat_dim",
    [
        (1, 54, 136, 24),
        (1, 29, 30, 6),
    ],
)
def test_graphmodel_output_shape(
    batch_size,
    n_atoms,
    atom_feat_dim,
    bond_feat_dim,
):
    model = models.NeuralGraphFingerprintModel(
        num_input_features=atom_feat_dim * 2 + bond_feat_dim,
        num_hidden_units=100,
        num_output_units=1,
    )

    atom_feats = torch.rand(batch_size, n_atoms, atom_feat_dim)
    bond_feats = torch.rand(batch_size, n_atoms, n_atoms, bond_feat_dim)
    adj_matrix = torch.rand(batch_size, n_atoms, n_atoms)

    output = model((atom_feats, bond_feats, adj_matrix))
    assert output.shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__])
