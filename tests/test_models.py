import pytest
import torch

from graphmodels import models


@pytest.mark.parametrize(
    "batch_size, n_atoms, n_bonds, atom_feat_dim, bond_feat_dim",
    [
        (4, 10, 12, 5, 3),
        (2, 7, 8, 8, 4),
    ],
)
def test_vanillanet_output_shape(
    batch_size,
    n_atoms,
    n_bonds,
    atom_feat_dim,
    bond_feat_dim,
):
    model = models.VanillaNet(atom_feat_dim, bond_feat_dim)

    atom_feats = torch.rand(batch_size, n_atoms, atom_feat_dim)
    bond_feats = torch.rand(batch_size, n_bonds, bond_feat_dim)

    output = model((atom_feats, bond_feats))
    assert output.shape == (batch_size,)


@pytest.mark.parametrize(
    "batch_size, n_atoms, atom_feat_dim, bond_feat_dim",
    [
        (64, 54, 136, 24),
        (64, 29, 30, 6),
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
