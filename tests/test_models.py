import pytest
import torch

from neuralfingerprint import models


@pytest.mark.parametrize(
    "batch_size, n_atoms, n_bonds, atom_feat_dim, bond_feat_dim",
    [
        (4, 10, 12, 5, 3),
        (2, 7, 8, 8, 4),
    ],
)
def test_vanillanet_output_shape(
    batch_size, n_atoms, n_bonds, atom_feat_dim, bond_feat_dim
):
    model = models.VanillaNet(atom_feat_dim, bond_feat_dim)

    atom_feats = torch.rand(batch_size, n_atoms, atom_feat_dim)
    bond_feats = torch.rand(batch_size, n_bonds, bond_feat_dim)

    output = model((atom_feats, bond_feats))
    assert output.shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__])
