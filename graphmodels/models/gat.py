import torch
import torch_scatter
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from torch import nn
from typeguard import typechecked as typechecker

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
        num_layers: int,
        num_heads: int,
        scaling: float,
        dropout: float,
        output_level: model_constants.OutputLevel | str,
    ):
        super().__init__()

        if output_level not in model_constants.ALLOWED_OUTPUT_LEVEL:
            raise ValueError(f"{output_level} isnt a valid output.")

        self.output_level = output_level

        gat_layers = [
            graph_attention_layers.MultiHeadGATLayer(
                n_node_features=n_node_features,
                n_hidden_features=n_hidden_features,
                num_heads=num_heads,
                dropout=dropout,
                scaling=scaling,
                apply_act=True,
            ),
        ]

        for i in range(num_layers - 1):
            gat_layers.append(
                graph_attention_layers.MultiHeadGATLayer(
                    n_node_features=n_hidden_features,
                    n_hidden_features=n_hidden_features,
                    num_heads=num_heads,
                    dropout=dropout,
                    scaling=scaling,
                    apply_act=True,
                ),
            )

        self.conv_layers = nn.ModuleList(gat_layers)

        self.output_layer = graph_attention_layers.MultiHeadGATLayer(
            n_node_features=n_hidden_features,
            n_hidden_features=n_out_channels,
            num_heads=1,
            dropout=dropout,
            scaling=scaling,
            apply_act=False,
        )

    def readout(
        self,
        x: Float[torch.Tensor, "nodes features"],
        batch_vector: Int[torch.Tensor, " batch"],
    ) -> Float[torch.Tensor, "out n_out_channels"]:
        if self.output_level == model_constants.OutputLevel.GRAPH:
            return torch_scatter.scatter_mean(
                src=x,
                index=batch_vector,
                dim=0,
            )

        return x

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
        batch_vector: Int[torch.Tensor, " batch"],
    ) -> Float[torch.Tensor, "out n_out_channels"]:
        for layer in self.conv_layers:
            node_features = layer(
                node_features=node_features,
                edge_index=edge_index,
            )

        out = self.output_layer(node_features, edge_index)
        return self.readout(out, batch_vector)


@jt(typechecker=typechecker)
class GATEdgeModel(nn.Module):
    """Implements a Graph Attention Network (GAT).

    This model consists of a stack of GAT layers with skip connections.
    Only node features are supported in this implementation.
    """

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        n_hidden_features: int,
        n_out_channels: int,
        num_layers: int,
        num_heads: int,
        scaling: float,
        dropout: float,
        output_level: model_constants.OutputLevel | str,
    ):
        super().__init__()

        if output_level not in model_constants.ALLOWED_OUTPUT_LEVEL:
            raise ValueError(f"{output_level} isnt a valid output.")

        self.output_level = output_level

        gat_layers = [
            graph_attention_layers.MultiHeadEdgeGATLayer(
                n_node_features=n_node_features,
                n_edge_features=n_edge_features,
                n_hidden_features=n_hidden_features,
                dropout=dropout,
                num_heads=num_heads,
                scaling=scaling,
                apply_act=True,
            ),
        ]

        for i in range(num_layers - 1):
            gat_layers.append(
                graph_attention_layers.MultiHeadEdgeGATLayer(
                    n_node_features=n_hidden_features,
                    n_edge_features=n_hidden_features,
                    n_hidden_features=n_hidden_features,
                    dropout=dropout,
                    num_heads=num_heads,
                    scaling=scaling,
                    apply_act=True,
                ),
            )

        self.conv_layers = nn.ModuleList(gat_layers)

        self.output_layer = graph_attention_layers.MultiHeadEdgeGATLayer(
            n_node_features=n_hidden_features,
            n_edge_features=n_hidden_features,
            n_hidden_features=n_out_channels,
            num_heads=1,
            dropout=dropout,
            scaling=scaling,
            apply_act=False,
        )

    def readout(
        self,
        x: Float[torch.Tensor, "nodes features"],
        batch_vector: Int[torch.Tensor, " batch"],
    ) -> Float[torch.Tensor, "out n_out_channels"]:
        if self.output_level == model_constants.OutputLevel.GRAPH:
            return torch_scatter.scatter_mean(
                src=x,
                index=batch_vector,
                dim=0,
            )

        return x

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
        batch_vector: Int[torch.Tensor, " batch"],
    ) -> Float[torch.Tensor, "out n_out_channels"]:
        for layer in self.conv_layers:
            node_features, edge_features = layer(
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index,
            )

        out, _ = self.output_layer(node_features, edge_features, edge_index)
        return self.readout(out, batch_vector)
