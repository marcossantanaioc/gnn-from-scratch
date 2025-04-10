import torch
from torch import nn


class VanillaNet(nn.Module):
    """
    A simple neural network that processes atom and bond features separately.

    This model is inspired by the neural fingerprint approach introduced by
    Duvenaud et al. (2015), but is not a direct implementation. Instead, it
    approximates the same core principles:
    - Featurizing atoms and bonds i ndividually
    - Applying learned transformations to each set
    - Aggregating (via mean pooling) to produce fixed-size vectors
    - Combining these vectors for property prediction

    The model uses fully connected layers for atom and bond processing,
    followed by concatenation, a non-linearity,and a final output layer for
    regression.
    """

    def __init__(
        self,
        n_atom_features: int,
        n_bond_features: int,
        n_out_features: int = 100,
    ):
        super().__init__()
        self.atom_layer = nn.Linear(n_atom_features, n_out_features)
        self.bond_layer = nn.Linear(n_bond_features, n_out_features)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(n_out_features * 2, 1)

    def forward(self, x):
        atom_features, bond_features = x
        atom_x = torch.mean(self.atom_layer(atom_features), dim=1)
        bond_x = torch.mean(self.bond_layer(bond_features), dim=1)

        concat_x = self.activation(torch.cat((atom_x, bond_x), dim=-1))

        return self.output_layer(concat_x).squeeze(-1)
