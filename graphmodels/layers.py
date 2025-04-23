import torch
from torch import nn

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
                modules.extend([nn.Linear(n_input_features, n_hidden_features), nn.ReLU()])
            else:
                modules.extend([nn.Linear(n_hidden_features, n_hidden_features), nn.ReLU()])
                
        modules.append(nn.Linear(n_hidden_features, n_node_features**2))

        self.edgelayer = nn.Sequential(*modules)

    def forward(self, x):
        edge_features, node_features, edge_index = x
        neighbors_index = edge_index[:, 1]

        edge_neighbors_features = edge_features[neighbors_index]
        node_features = node_features[neighbors_index]

        edge_out = self.edgelayer(edge_neighbors_features)

        message = edge_out.view(-1, node_features.size(-1), node_features.size(-1))

        return (message @ node_features.unsqueeze(-1)).squeeze(-1)