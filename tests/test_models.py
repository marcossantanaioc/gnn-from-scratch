import pytest
import torch

from neuralfingerprint import models


def test_vanillanet_output_shape():
    model = models.VanillaNet(n_atom_features=5, n_bond_features=3)
    batch_size = 8
    atom_feats = torch.rand(batch_size, 10, 5)  # [B, N_atoms, atom_feat_dim]
    bond_feats = torch.rand(batch_size, 12, 3)  # [B, N_bonds, bond_feat_dim]

    output = model((atom_feats, bond_feats))
    assert output.shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__])
