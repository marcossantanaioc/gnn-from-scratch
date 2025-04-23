import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from graphmodels import layers


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
            ]
        )
        self.o = nn.ModuleList(
            [nn.Linear(n_hidden_units, n_hidden_units) for r in range(radius)]
        )
        self.output_layer = nn.Linear(n_hidden_units, n_output_units)
        self.radius = radius

    def forward(self, x):
        atom_feats, adj_matrix = x
        f = []

        # Iterate over radius. Every pass we collect messages from more distant neighbors up to self.radius
        for r in range(self.radius):
            # Fetch neighbors messages. Non-neighbors are zeroed while neighbors are summed
            neighbors_features = adj_matrix @ atom_feats

            # Pass messages to every atom
            v = atom_feats + neighbors_features

            # Update atom's features with a hidden layer and non-linearity
            ra = F.tanh(self.h[r](v))

            # Make a sparse representation of the fingerprint, where the softmax creates a prob distribution over features
            i = F.softmax(self.o[r](ra), dim=-1)

            # Update atom features to the new features
            atom_feats = ra

            # Add the fingerprint to the list. We sum over all atoms to get a representation of features for that molecule [n_hidden_units]
            f.append(i.sum(dim=1))

        # Sum over layers to get the final fingerprint for a molecule

        # print(torch.stack(f).shape)
        fp = torch.stack(f).sum(dim=0)

        return self.output_layer(fp)


class MPNNv1(nn.Module):
    """
    A Message Passing Neural Network (MPNN) model for graph-level prediction.

    This model implements a standard MPNN architecture consisting of three main phases:
    message passing, node update, and readout. It processes graph structures with
    node and edge (bond) features to produce a graph-level output.

    Args:
        n_node_features (int): The dimensionality of the input node features.
        n_bond_features (int): The dimensionality of the input bond (edge) features.
        n_bond_hidden_features (int, optional): The number of hidden features in the
            message passing layers. Defaults to 200.
        n_hidden_features (int, optional): The number of hidden features in the
            node update and readout layers. Defaults to 200.
        n_message_passes (int, optional): The number of message passing iterations
            (layers) in the edge layer. Defaults to 3.
        n_update_layers (int, optional): The number of layers in the node update
            network. Defaults to 2.
        n_readout_steps (int, optional): The number of steps (layers) in the
            readout network. Defaults to 2.
        n_out_features (int): The dimensionality of the final graph-level output.

    Inputs:
        x (tuple): A tuple containing the following tensors:
            - edge_features (torch.Tensor): Tensor of edge features, shape [num_edges, n_bond_features].
            - node_features (torch.Tensor): Tensor of node features, shape [num_nodes, n_node_features].
            - edge_index (torch.Tensor): Graph connectivity in COO format, shape [2, num_edges].
            - batch_vector (torch.Tensor): Batch assignment vector for nodes, shape [num_nodes].
              This is used for aggregating node features to obtain graph-level
              representations for batched graphs.

    Returns:
        torch.Tensor: The graph-level output tensor, shape [num_graphs, n_out_features].
    """
    def __init__(
        self,
        n_node_features: int,
        n_bond_features: int,
        n_out_features: int,
        *,
        n_bond_hidden_features: int = 200,
        n_hidden_features: int = 200,
        n_message_passes: int = 3,
        n_update_layers: int = 2,
        n_readout_steps: int = 2,
    ):
        super().__init__()

        self.edge_layer = layers.EdgeLayer(
            n_input_features=n_bond_features,
            n_hidden_features=n_bond_hidden_features,
            n_node_features=n_node_features,
            passes=n_message_passes,
        )
        self.update_layer = layers.UpdateLayer(
            n_input_features=n_node_features,
            n_hidden_features=n_hidden_features,
            n_node_features=n_node_features,
            num_layers=n_update_layers,
        )
        self.readout_layer = layers.ReadoutLayer(
            n_input_features=n_node_features,
            n_hidden_features=n_hidden_features,
            n_out_features=n_out_features,
            num_layers=n_readout_steps,
        )

    def forward(self, x):
        edge_features, node_features, edge_index, batch_vector = x

        messages = self.edge_layer((edge_features, node_features, edge_index))

        updated_nodes = self.update_layer(
            (messages, node_features, edge_index)
        )

        readout = self.readout_layer((updated_nodes, batch_vector))

        return readout
