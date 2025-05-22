import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class NeuralGraphFingerprintModel(nn.Module):
    """Neural Graph Fingerprint model.
    This model is inspired by the neural fingerprint approach introduced by
    Duvenaud et al. (2015). We implemented the pseudocode version, not the
    github code. This means the model only uses atom features as inputs.
    """

    def __init__(
        self,
        n_input_features: int,
        n_hidden_units: int,
        n_output_units: int,
        radius: int,
    ):
        super().__init__()

        self.h = nn.ModuleList(
            [
                nn.Linear(
                    n_input_features if r == 0 else n_hidden_units,
                    n_hidden_units,
                )
                for r in range(radius)
            ],
        )
        self.o = nn.ModuleList(
            [nn.Linear(n_hidden_units, n_hidden_units) for r in range(radius)],
        )
        self.output_layer = nn.Linear(n_hidden_units, n_output_units)
        self.radius = radius

    def forward(self, x):
        atom_feats, adj_matrix = x
        f = []

        # Iterate over radius.
        for r in range(self.radius):
            # Fetch neighbors messages.
            neighbors_features = adj_matrix @ atom_feats

            # Pass messages to every atom
            v = atom_feats + neighbors_features

            # Update atom's features with a hidden layer and non-linearity
            ra = F.tanh(self.h[r](v))

            # Make a sparse representation of the fingerprint
            i = F.softmax(self.o[r](ra), dim=-1)

            # Update atom features to the new features
            atom_feats = ra

            # Add the fingerprint to the list.
            f.append(i.sum(dim=1))

        # Sum over layers to get the final fingerprint for a molecule

        # print(torch.stack(f).shape)
        fp = torch.stack(f).sum(dim=0)

        return self.output_layer(fp)
