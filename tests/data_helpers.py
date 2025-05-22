"""Helper functions to generate batches for unittests."""

import dataclasses
import random
from collections.abc import Sequence

import torch
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from typeguard import typechecked as typechecker

from graphmodels import constants


@jt(typechecker=typechecker)
@dataclasses.dataclass(kw_only=True, frozen=True)
class SampleEntry:
    """Stores features and adjacency information for a test graph."""

    adj_matrix: Int[torch.Tensor, "nodes nodes"]
    edge_features: Float[torch.Tensor, "edges edge_features"]
    node_features: Float[torch.Tensor, "nodes node_features"]
    edge_indices: Int[torch.Tensor, "2 _"]
    target: Float[torch.Tensor, ""]
    total_nodes: int
    total_edges: int


@jt(typechecker=typechecker)
@dataclasses.dataclass(kw_only=True, frozen=True)
class SampleBatch:
    """Stores datasets in a batch."""

    dsets: Sequence[SampleEntry]
    total_nodes: int
    total_edges: int


def _create_random_graph(
    min_num_nodes: int,
    max_num_nodes: int,
    n_node_features: int,
    n_edge_features: int,
):
    n_nodes = random.randint(min_num_nodes, max_num_nodes)

    target = torch.rand(1).squeeze(-1)

    adj_matrix = torch.rand(n_nodes, n_nodes)
    adj_matrix[adj_matrix > 0.5] = 1
    adj_matrix[adj_matrix <= 0.5] = 0

    # Make symmetric
    adj_matrix = adj_matrix + adj_matrix.T  # /2

    # Remove self-loops
    adj_matrix.fill_diagonal_(0)

    adj_matrix = torch.where(adj_matrix >= 1, 1, 0)

    edge_indices = torch.nonzero(adj_matrix).T

    node_features = torch.rand(n_nodes, n_node_features)
    edge_features = torch.rand(edge_indices.size(1), n_edge_features)

    return SampleEntry(
        adj_matrix=adj_matrix,
        edge_features=edge_features,
        node_features=node_features,
        edge_indices=edge_indices,
        target=target,
        total_nodes=n_nodes,
        total_edges=int(adj_matrix.sum().item()),
    )


def _generate_random_dataset(
    min_num_nodes: int = 2,
    max_num_nodes: int = 50,
    n_node_features: int = constants.NUM_NODE_FEATURES,
    n_edge_features: int = constants.NUM_EDGE_FEATURES,
    num_examples: int = 10,
) -> SampleBatch:
    dsets = []
    total_nodes = 0
    total_edges = 0

    for _ in range(num_examples):
        dset = _create_random_graph(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            min_num_nodes=min_num_nodes,
            max_num_nodes=max_num_nodes,
        )
        if dset.edge_indices.numel() > 0:
            total_nodes += dset.total_nodes
            total_edges += dset.total_edges
            dsets.append(dset)

    return SampleBatch(
        dsets=dsets,
        total_nodes=total_nodes,
        total_edges=total_edges,
    )
