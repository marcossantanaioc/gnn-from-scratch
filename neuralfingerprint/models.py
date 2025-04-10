import torch
import torch.nn.functional as F  # noqa: N812
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

    We implement a simplified approximation of the Neural Graph Fingerprint
    model by Duvenaud et al., where atom and bond-level features are
    independently processed through feedforward layers and aggregated to create
    a graph-level representation.Unlike the original NGFP, this model does not
    incorporate iterative message-passing or neighborhood-based convolutions,
    and therefore serves primarily as a baseline or ablation.

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


class NeuralGraphFingerprintModel(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        num_hidden_units: int,
        num_output_units: int,
    ):
        super().__init__()

        self.lin_layer = nn.Linear(num_input_features, num_hidden_units)
        self.out_layer = nn.Linear(num_hidden_units, num_output_units)

    def forward(self, x):
        atom_feats, bond_feats, adj_matrix = x

        atom_feat_expanded = atom_feats.unsqueeze(1).expand(
            -1,
            atom_feats.size()[1],
            -1,
            -1,
        )

        # Concat features. This avoids iterating over each atom.
        all_feats = torch.cat([atom_feat_expanded, bond_feats], dim=-1)

        # Mask features using the adj. matrix.
        # We zero-out features that are not associated with neighboring atoms.
        neighbor_messages = all_feats * adj_matrix.unsqueeze(-1)

        # Aggregate neighbor information (e.g. sum over neighbors)
        aggregated = neighbor_messages.sum(
            dim=1,
        )  # shape: [N_atoms, message_dim]

        # Update atom features
        out = F.relu(
            self.lin_layer(torch.cat([atom_feats, aggregated], dim=-1)),
        )
        # Output molecule-level prediction
        return self.out_layer(out).sum(dim=1)
