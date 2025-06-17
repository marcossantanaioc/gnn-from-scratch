import torch
import torch.nn.functional as F  # noqa: N812
import torch_scatter
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from torch import nn
from typeguard import typechecked as typechecker


@jt(typechecker=typechecker)
class G2(nn.Module):
    """Implementation of the gating mechanism described in the paper
    GRADIENT GATING FOR DEEP MULTI-RATE LEARNING ON GRAPHS
    (Rusch et al, 2022)
    The gating mechanism is based on the modulation of update rates of node
    features to avoid oversmoothing and allow the construction of
    deeper architetures while keeping performance.

    Attributes:
        conv: a GNN layer used in the computation of update gradients.
        p: exponential applied to compute the update gradients. Default: 2

    """

    def __init__(self, conv: nn.Module, p: int = 2):
        super().__init__()
        self.p = p
        self.conv = conv

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> Float[torch.Tensor, "edges node_features"]:
        x = self.conv(node_features, edge_index)
        target_nodes = edge_index[1]
        neighbor_nodes = edge_index[0]
        diffs = torch.abs(x[neighbor_nodes] - x[target_nodes]).pow(self.p)
        tau = F.tanh(
            torch_scatter.scatter_mean(
                diffs,
                index=target_nodes,
                dim=0,
                dim_size=node_features.size(0),
            ),
        )
        return tau
