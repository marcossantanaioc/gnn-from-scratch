import torch
import torch.nn.functional as F  # noqa: N812
import torch_scatter
from torch import nn


class SimpleGAT(nn.Module):
    """Implements a simple graph attention layer.

    Attributes
        n_node_features: number of input node features.
        n_hidden_features: number of hidden features in intermediate layers.
        scaling: scaling constant for LeakyRelu
    """

    def __init__(
        self,
        n_node_features: int,
        n_hidden_features: int,
        scaling: float = 0.2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.scaling = scaling
        self.dropout = dropout
        self.w = nn.Linear(n_node_features, n_hidden_features)
        self.attn = nn.Linear(n_hidden_features * 2, 1)

    def compute_attention(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Computes attention score between nodes i and j.

        The attention mechanism is defined as:

            α_ij = softmax_j( LeakyReLU( aᵀ [ W h_i || W h_j ] ) )

        where:
        - h_i, h_j are the input features of nodes i and j,
        - W is a shared learnable weight matrix,
        - a is a learnable attention vector,
        - || denotes vector concatenation,
        - softmax_j is applied over all neighbors j ∈ N(i) of node i.

        The updated feature for node i is computed as:

            h_i' = σ( Σ_{j ∈ N(i)} α_ij · W h_j )

        where σ is a non-linear activation function (e.g., ELU or ReLU).

        Args:
            node_features: input node features (shape = N, F)
            where N is the number of nodes and F the number of features.
            edge_index: a matrix where every row corresponds to an edge.
        Returns:
            Attention scores for nodes.
        """

        h = F.leaky_relu(self.w(node_features), self.scaling)

        neighbors_nodes = edge_index[1]
        target_nodes = edge_index[0]

        h_i = h[target_nodes]
        h_j = h[neighbors_nodes]
        h_concat = torch.cat([h_i, h_j], dim=-1)

        eij = F.dropout(self.attn(h_concat), self.dropout)

        attention_score = torch_scatter.scatter_softmax(
            src=eij,
            index=target_nodes,
            dim=0,
        )

        return (attention_score * h_j), h, target_nodes

    def forward(self, node_features, edge_index):
        message, node_features, target_nodes = self.compute_attention(
            node_features=node_features,
            edge_index=edge_index,
        )

        out = torch_scatter.scatter_add(
            message,
            target_nodes,
            dim=0,
            dim_size=node_features.size(0),
        )
        return F.leaky_relu(out, self.scaling)
