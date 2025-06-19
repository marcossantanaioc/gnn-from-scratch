"""Graph attention layers"""

import torch
import torch.nn.functional as F  # noqa: N812
import torch_scatter
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from torch import nn
from typeguard import typechecked as typechecker


@jt(typechecker=typechecker)
class GINLayer(nn.Module):
    """Implementation of a graph isomorphism network

    This class implements the Graph Isomorphism Network (GIN) architecture
    proposed by Xu et al.

    The key idea behind GIN is to replace average or max aggregation with
    summation over neighbor features.
    This enables the model to match the discriminative power of the
    Weisfeiler-Lehman (1-WL) test, making it
    more expressive for distinguishing non-isomorphic graphs.

    Attributes:
        n_input_features: number of input features for the MLP
        n_hidden_features: number of hidden features in the MLP
        dropout: amount of dropout regularization to add
        add_skip_connection: whether to add initial node features to updated
        node features.

    """

    def __init__(
        self,
        n_input_features: int,
        n_hidden_features: int,
        dropout: float = 0.0,
        add_skip_connection: bool = False,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_features),
            nn.ReLU(),
            nn.Linear(n_hidden_features, n_hidden_features),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=dropout)

        self.eta = nn.Parameter(torch.empty(1, 1))
        nn.init.xavier_uniform_(self.eta)

        self.norm = nn.LayerNorm(n_hidden_features)

        self.add_skip_connection = add_skip_connection

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> Float[torch.Tensor, "nodes node_features"]:
        initial_features = node_features

        hu = node_features[edge_index[0]]

        sum_hu = torch_scatter.scatter_add(
            src=hu,
            index=edge_index[1],
            dim=0,
            dim_size=node_features.size(0),
        )

        node_features = F.relu(
            self.mlp(((1 + self.eta) * node_features) + sum_hu),
        )

        node_features = self.norm(self.dropout(node_features))

        if self.add_skip_connection:
            node_features = node_features + initial_features

        return node_features
