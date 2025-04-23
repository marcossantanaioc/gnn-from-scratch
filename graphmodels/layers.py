import torch
from torch import nn
import torch.nn.functional as F


class EdgeLayer(nn.Module):
    """Implements the Edge network as described in Gilmer et al.'s MPNN framework.

    The Edge Network module transforms the features of neighboring nodes by
    incorporating information from the edges connecting them. For a given node v,
    it gathers the features of its connecting edges (e_vw) and its neighbor nodes
    (h_w). The edge features e_vw are used to generate a transformation matrix,
    which is then applied to the neighbor node features h_w. This process
    effectively
    conditions the transformation of node features on the properties of the
    interconnecting edges.
    """

    def __init__(
        self,
        n_input_features: int,
        n_hidden_features: int,
        n_node_features: int,
        passes: int = 3,
    ):
        super().__init__()
        modules = []
        for i in range(passes):
            if i == 0:
                modules.extend(
                    [nn.Linear(n_input_features, n_hidden_features), nn.ReLU()]
                )
            else:
                modules.extend(
                    [
                        nn.Linear(n_hidden_features, n_hidden_features),
                        nn.ReLU(),
                    ]
                )

        modules.append(nn.Linear(n_hidden_features, n_node_features**2))

        self.edgelayer = nn.Sequential(*modules)

    def forward(self, x):
        edge_features, node_features, edge_index = x
        neighbors_index = edge_index[:, 1]

        edge_neighbors_features = edge_features[neighbors_index]
        node_features = node_features[neighbors_index]

        edge_out = self.edgelayer(edge_neighbors_features)

        message = edge_out.view(
            -1, node_features.size(-1), node_features.size(-1)
        )

        return (message @ node_features.unsqueeze(-1)).squeeze(-1)


class UpdateLayer(nn.Module):
    """Implements the node update mechanism inspired by the Gated Graph
    Sequence Neural Networks (GG-NN) paper (Li et al., 2016).

    This layer updates the feature vectors of each node in the graph by
    aggregating incoming messages from its neighbors and using a Gated
    Recurrent Unit (GRU) to combine these messages with the node's current
    features.

    Args:
        n_input_features (int): The dimensionality of the incoming messages
            (which should match the dimensionality of the aggregated messages).
        n_hidden_features (int): The dimensionality of the hidden state within
            the GRU, which will also be the dimensionality of the updated node
            features before the final output layer.
        n_node_features (int): The dimensionality of the output node features
            after the linear output layer.
        num_layers (int, optional): The number of layers in the GRU.
            Defaults to 3.

    Inputs:
        x (tuple of torch.Tensor): A tuple containing the following tensors:
            - messages (torch.Tensor of shape (E, F_msg)): Tensor containing
              the messages passed along the edges, where E is the number of
              edges and F_msg is the dimensionality of each message
              (should be equal to n_input_features).
            - node_features (torch.Tensor of shape (N, F_node)): Tensor
              containing the current features of each node in the graph,
              where N is the number of nodes and F_node is the dimensionality
              of the node features.
            - edge_index (torch.Tensor of shape (2, E)): Tensor defining the
              edges in the graph in COO format, where each column (u, v)
              represents an edge from source node u to target node v.

    Outputs:
        torch.Tensor of shape (N, n_node_features): Tensor containing the
        updated features for each node in the graph after passing through the
        GRU and the final linear layer with ReLU activation.
    """

    def __init__(
        self,
        n_input_features: int,
        n_hidden_features: int,
        n_node_features: int,
        num_layers: int = 3,
    ):
        super().__init__()
        self.update_layers = nn.GRU(
            n_input_features, n_hidden_features, num_layers=num_layers
        )
        self.output_layer = nn.Linear(n_hidden_features, n_node_features)

    def forward(self, x):
        messages, node_features, edge_index = x
        source_nodes = edge_index[:, 0]

        aggregated_messages = torch.zeros_like(node_features)
        aggregated_messages.scatter_add_(
            0,
            source_nodes.unsqueeze(-1).expand(-1, messages.size(-1)),
            messages,
        )
        updated_nodes, _ = self.update_layers(
            aggregated_messages + node_features
        )

        return self.output_layer(F.relu(updated_nodes))
