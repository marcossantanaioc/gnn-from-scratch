import torch
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from torch import nn
from typeguard import typechecked as typechecker

from graphmodels.layers import mpnn_layers


@jt(typechecker=typechecker)
class MPNNv1(nn.Module):
    """
    A Message Passing Neural Network (MPNN) model for graph-level prediction.

    This model implements a standard MPNN architecture consisting of 3 steps:
    message passing, node update, and readout. It processes graph structures w/
    node and edge (bond) features to produce a graph-level output.

    Args:
        n_node_features: The dimensionality of the input node features.
        n_edge_features: The dimensionality of the input bond (edge) features.
        n_edge_hidden_features: The number of hidden features in the
            message passing mpnn_layers. Defaults to 200.
        n_hidden_features: The number of hidden features in the
            node update and readout mpnn_layers. Defaults to 200.
        n_message_passes: The number of message passing iterations
            (layers) in the edge layer. Defaults to 3.
        n_update_layers: The number of layers in the node update
            network. Defaults to 2.
        n_readout_steps: The number of steps (layers) in the
            readout network. Defaults to 2.
        n_out_features: The dimensionality of the final graph-level output.

    Inputs:
        x (tuple): A tuple containing the following tensors:
            - edge_features: Tensor of edge features.
            - node_features: Tensor of node features.
            - edge_index: Graph connectivity in COO format.
            - batch_vector: Batch assignment vector for nodes.
              This is used for aggregating node features to obtain graph-level
              representations for batched graphs.

    Returns:
        torch.Tensor: The graph-level output tensor
    """

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        n_out_features: int,
        *,
        n_hidden_features: int = 200,
        n_towers: int = 8,
        n_readout_steps: int = 2,
        n_update_steps: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.update_layer = mpnn_layers.MessagePassingLayer(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_update_steps=n_update_steps,
            n_towers=n_towers,
            dropout=dropout,
        )
        self.readout_layer = mpnn_layers.ReadoutLayer(
            n_node_features=n_node_features,
            n_hidden_features=n_hidden_features,
            n_out_features=n_out_features,
            num_layers=n_readout_steps,
        )

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
        batch_vector: Int[torch.Tensor, " batch"],
    ) -> Float[torch.Tensor, "out 1"]:
        updated_nodes = self.update_layer(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )

        readout = self.readout_layer(
            node_features=updated_nodes,
            batch_vector=batch_vector,
        )

        return readout
