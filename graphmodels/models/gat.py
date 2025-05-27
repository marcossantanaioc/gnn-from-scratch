import torch
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from torch import nn
from typeguard import typechecked as typechecker

from graphmodels.layers import constants as layer_constants
from graphmodels.layers import graph_attention_layers
from graphmodels.models import constants as model_constants


@jt(typechecker=typechecker)
class GATModel(nn.Module):
    """Implements a Graph Attention Network (GAT).

    This model consists of a stack of GAT layers with skip connections.
    Only node features are supported in this implementation.
    """

    def __init__(
        self,
        n_node_features: int,
        n_hidden_features: int,
        n_out_channels: int,
        n_layers: int,
        scaling: float,
        dropout: float,
        agg_method: layer_constants.PoolingMethod | str,
        output_level: model_constants.OutputLevel | str,
    ):
        super().__init__()

        if agg_method not in model_constants.ALLOWED_POOLING:
            raise ValueError(f"{agg_method} isnt a valid pooling method.")

        if output_level not in model_constants.ALLOWED_OUTPUT_LEVEL:
            raise ValueError(f"{output_level} isnt a valid output.")

        self.output_level = output_level

        gat_layers = [
            graph_attention_layers.GraphAttentionLayerSkip(
                n_node_features=n_node_features,
                n_hidden_features=n_hidden_features,
                scaling=scaling,
                dropout=dropout,
            ),
        ]

        for _ in range(1, n_layers):
            gat_layers.append(
                graph_attention_layers.GraphAttentionLayerSkip(
                    n_node_features=n_hidden_features,
                    n_hidden_features=n_hidden_features,
                    scaling=scaling,
                    dropout=dropout,
                ),
            )
        self.gat = nn.ModuleList(gat_layers)
        self.output_layer = nn.Linear(n_hidden_features, n_out_channels)

    def readout(
        self,
        x: Float[torch.Tensor, "nodes features"],
        batch_vector: Int[torch.Tensor, " batch"],
    ) -> Float[torch.Tensor, "out n_out_channels"]:
        emb_dim = x.size(-1)
        num_batches = int(batch_vector.max()) + 1
        if self.output_level == model_constants.OutputLevel.GRAPH:
            mol_embeddings = torch.zeros(
                num_batches,
                emb_dim,
                device=x.device,
            )

            mol_embeddings.index_add_(0, batch_vector, x)

            return self.output_layer(mol_embeddings)

        return self.output_layer(x)

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
        batch_vector: Int[torch.Tensor, " batch"],
    ) -> Float[torch.Tensor, "out n_out_channels"]:
        for layer in self.gat:
            node_features = layer(
                node_features=node_features,
                edge_index=edge_index,
            )

        readout = self.readout(x=node_features, batch_vector=batch_vector)
        return readout
