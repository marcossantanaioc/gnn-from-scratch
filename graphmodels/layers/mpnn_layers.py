import torch
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from torch import nn
from typeguard import typechecked as typechecker


@jt(typechecker=typechecker)
class MultiTowerEdge(nn.Module):
    """Implements multitower edge network as described by Gilmer et al."""

    def __init__(
        self,
        n_edge_features: int,
        n_node_features: int,
        n_towers: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.n_towers = n_towers
        self.tower_dimension = n_node_features // n_towers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features

        self.edgetower = nn.Sequential(
            nn.Linear(
                n_edge_features,
                self.n_node_features * self.tower_dimension,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ):
        neighbors_index = edge_index[1]

        neighbors_node_features = node_features[neighbors_index]

        edge_out = self.edgetower(edge_features).view(
            edge_features.size(0),
            self.n_towers,
            self.tower_dimension,
            -1,
        )

        neighbors_node_features_reshaped = neighbors_node_features.view(
            -1,
            self.n_towers,
            self.tower_dimension,
        ).unsqueeze(-1)

        return (edge_out @ neighbors_node_features_reshaped).view(
            -1,
            self.n_node_features,
        )


@jt(typechecker=typechecker)
class EdgeLayer(nn.Module):
    """Implements the Edge network as described in Gilmer et al.'s MPNN.

    The Edge Network module transforms the features of neighboring nodes by
    incorporating information from the edges connecting them.
    For a given node v, it gathers the features of its connecting edges (e_vw)
    and its neighbor nodes (h_w).
    The edge features e_vw are used to generate a transformation matrix,
    which is then applied to the neighbor node features h_w. This process
    effectively
    conditions the transformation of node features on the properties of the
    interconnecting edges.
    """

    def __init__(
        self,
        n_edge_features: int,
        n_edge_hidden_features: int,
        n_node_features: int,
        n_update_steps: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        modules = []
        for i in range(n_update_steps):
            if i == 0:
                modules.extend(
                    [
                        nn.Linear(n_edge_features, n_edge_hidden_features),
                        nn.Dropout(dropout),
                        nn.ReLU(),
                    ],
                )
            else:
                modules.extend(
                    [
                        nn.Linear(
                            n_edge_hidden_features,
                            n_edge_hidden_features,
                        ),
                        nn.Dropout(dropout),
                        nn.ReLU(),
                    ],
                )

        modules.append(nn.Linear(n_edge_hidden_features, n_node_features**2))

        self.edgelayer = nn.Sequential(*modules)

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ):
        neighbors_index = edge_index[1]

        neighbors_node_features = node_features[neighbors_index]

        edge_out = self.edgelayer(edge_features)

        message = edge_out.view(
            -1,
            neighbors_node_features.size(-1),
            neighbors_node_features.size(-1),
        )

        return (message @ neighbors_node_features.unsqueeze(-1)).squeeze(-1)


@jt(typechecker=typechecker)
class MessagePassingLayer(nn.Module):
    """Implements a Gated Graph Neural Networks (GG-NN).

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
        n_node_features: int,
        n_edge_features: int,
        n_towers: int = 8,
        n_update_steps: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.edge_layer = MultiTowerEdge(
            n_edge_features=n_edge_features,
            n_node_features=n_node_features,
            n_towers=n_towers,
            dropout=dropout,
        )

        self.update_cell = nn.GRUCell(n_node_features, n_node_features)

        self.n_update_steps = n_update_steps

        self.output_layer = nn.Sequential(
            nn.Linear(n_node_features, n_node_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_node_features, n_node_features),
        )

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ):
        for _ in range(self.n_update_steps):
            # Collect messages
            messages = self.edge_layer(
                edge_features=edge_features,
                node_features=node_features,
                edge_index=edge_index,
            )

            target_nodes = edge_index[0]

            # Aggregate messages
            aggregated_messages = torch.zeros_like(node_features)
            aggregated_messages.index_add_(0, target_nodes, messages)
            node_features = self.update_cell(
                aggregated_messages,
                node_features,
            )

        return self.output_layer(node_features)


@jt(typechecker=typechecker)
class ReadoutLayer(nn.Module):
    """Aggregates node features to obtain a graph-level representation
    suitable for graph-level tasks within a Message Passing Neural Network
    (MPNN) framework.

    This layer first applies a multi-layer perceptron (MLP) to each node's
    feature vector. Then, it aggregates these transformed node features for
    each graph in the batch using summation to produce a single
    embedding vector representing the entire graph. Finally, it applies a
    linear layer to this graph embedding to obtain the output for the
    graph-level task.

    Args:
        n_input_features (int): The dimensionality of the input node features.
        n_hidden_features (int): The dimensionality of the hidden layers within
            the MLP applied to each node.
        n_out_features (int): The dimensionality of the final output graph
            embeddings.
        num_layers (int, optional): The number of hidden layers in the MLP
            applied to each node. Defaults to 2.

    Inputs:
        x (tuple of torch.Tensor): A tuple containing the following tensors:
            - updated_node_features (torch.Tensor of shape (N, F_node)): Tensor
              containing the final feature vectors for all nodes in the batch
              after message passing and update steps, where N is the total
              number of nodes and F_node is the dimensionality of the node
              features (should match n_input_features).
            - batch_vector (torch.Tensor of shape (N,)): A 1D tensor where each
              element indicates the graph index (within the batch) to which the
              corresponding node belongs.

    Outputs:
        torch.Tensor of shape (B, n_out_features): Tensor containing the
        graph-level embeddings for each graph in the batch, where B is the
        number of graphs in the batch (determined by the maximum value in
        `batch_vector` + 1).
    """

    def __init__(
        self,
        n_node_features: int,
        n_hidden_features: int,
        n_out_features: int,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend(
                    [nn.Linear(n_node_features, n_hidden_features), nn.ReLU()],
                )
            else:
                layers.extend(
                    [
                        nn.Linear(n_hidden_features, n_hidden_features),
                        nn.Dropout(dropout),
                        nn.ReLU(),
                    ],
                )

        self.readout = nn.Sequential(*layers)
        self.output_layer = nn.Linear(n_hidden_features, n_out_features)

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        batch_vector: Int[torch.Tensor, " batch"],
    ):
        readout = self.readout(node_features)

        num_batches = int(batch_vector.max()) + 1
        emb_dim = readout.size(-1)

        mol_embeddings = torch.zeros(
            num_batches,
            emb_dim,
            device=readout.device,
        )

        mol_embeddings.index_add_(0, batch_vector, readout)

        return self.output_layer(mol_embeddings)
