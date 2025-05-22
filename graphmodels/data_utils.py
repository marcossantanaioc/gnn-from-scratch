"""Utility functions to generate batches."""

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from torch.nn import functional as F  # noqa: N812
from typeguard import typechecked as typechecker

from graphmodels.datasets import mpnn_dataset, ngf_dataset


@jt(typechecker=typechecker)
@dataclasses.dataclass(kw_only=True, frozen=True)
class NeuralGraphFingerprintBatch:
    """Store batch information for the neural graph fingerprint model."""

    adj_matrix: Int[torch.Tensor, "nodes nodes"]
    edge_index: Int[torch.Tensor, "2 edges"]
    node_features: Float[torch.Tensor, "nodes node_features"]
    targets: Float[torch.Tensor, " ground_truth"]

    def to_dict(self) -> dict[str, Any]:
        """Generates a dictionary from fields."""
        return dataclasses.asdict(self)


@jt(typechecker=typechecker)
@dataclasses.dataclass(kw_only=True, frozen=True)
class MPNNBatch:
    """Store batch information for the MPNNv1 model."""

    edge_index: Int[torch.Tensor, "2 edges"]
    batch_vector: Int[torch.Tensor, " batch"]
    node_features: Float[torch.Tensor, "nodes node_features"]
    edge_features: Float[torch.Tensor, "edges edge_features"]
    targets: Float[torch.Tensor, " ground_truth"]

    def to_dict(self) -> dict[str, Any]:
        """Generates a dictionary from fields."""
        return dataclasses.asdict(self)


@jt(typechecker=typechecker)
def create_batch_edge_index(
    edge_indices: Sequence[Int[torch.Tensor, "2 ..."]],
) -> Int[torch.Tensor, "2 ..."]:
    """Creates a batch of edge indices.

    This function takes all edge index matrices in the data,
    and collates them into a single matrix of shape [2, N],
    where N is the total number of edges in the dataset.
    Nodes are offset by the total number of nodes from the
    previous step. This implementation aims to mimic Torch
    geometric batching.

    Args:
        edge_indices: a collection of edge index matrix for the batch.

    Returns:
        A concatenation of all edge indices.
    """

    offset = 0
    to_concat = []
    total_num_nodes = 0
    for edge_index in edge_indices:
        num_nodes = int(edge_index.max() + 1)

        edge_index_new = edge_index + offset

        to_concat.append(edge_index_new)
        offset += num_nodes
        total_num_nodes += num_nodes

    return torch.cat(to_concat, dim=-1)


@jt(typechecker=typechecker)
def mpnn_collate_diag(batch: list[mpnn_dataset.MPNNEntry]) -> MPNNBatch:
    """Generates a batch of inputs for MPNN."""
    # Build batch vector
    num_atoms_per_mol = [nf.node_features.size(0) for nf in batch]
    batch_vector = torch.cat(
        [
            torch.full(size=(n,), fill_value=i, dtype=torch.long)
            for i, n in enumerate(num_atoms_per_mol)
        ],
    )  # [total_atoms], values in [0, batch_size-1]

    all_edge_indices = create_batch_edge_index([x.edge_indices for x in batch])
    all_node_features = torch.concat([x.node_features for x in batch], dim=0)
    all_edge_features = torch.concat([x.edge_features for x in batch], dim=0)
    targets = torch.stack([x.target for x in batch])

    return MPNNBatch(
        batch_vector=batch_vector,
        edge_index=all_edge_indices,
        node_features=all_node_features,
        edge_features=all_edge_features,
        targets=targets,
    )


@jt(typechecker=typechecker)
def neuralgraph_collate_diag(batch: list[ngf_dataset.NeuralFingerprintEntry]):
    """Generates a batch of inputs for NeuralGraphFingerprint."""
    # Build batch vector
    num_atoms_per_mol = [nf.node_features.size(0) for nf in batch]
    batch_vector = torch.cat(
        [
            torch.full((n,), i, dtype=torch.long)
            for i, n in enumerate(num_atoms_per_mol)
        ],
    )  # [total_atoms], values in [0, batch_size-1]

    all_adj_matrix = torch.block_diag(*[x.adj_matrix for x in batch])
    all_node_features = torch.concat([x.node_features for x in batch], dim=0)
    targets = torch.stack([x.target for x in batch])

    return NeuralGraphFingerprintBatch(
        adj_matrix=all_adj_matrix,
        edge_index=batch_vector,
        node_features=all_node_features,
        targets=targets,
    )


@jt(typechecker=typechecker)
def neuralgraph_longest_collate(
    batch: list[ngf_dataset.NeuralFingerprintEntry],
    max_num_atoms: int,
):
    """
    Collate function that pads each graph in the batch to match the size of
    the largest graph.

    This function processes a list of graphs, each potentially with a different
    number of nodes, and pads their features so that all graphs in the batch
    have the same number of nodes.
    Padding is applied to match the size of the largest graph in the batch.

    Args:
        batch: a collection of tuples containing the features and targets.
            for all elements in a batch.

    Returns:
        concatenated features and targets for all elements in a batch.
    """
    all_node_features = []
    all_edge_features = []
    all_adj_matrices = []
    all_targets = []

    # Get max number of atoms in data
    for entry in batch:
        num_to_pad = max_num_atoms - entry.node_features.shape[0]
        node_feats_padded = F.pad(
            entry.node_features,
            pad=(0, 0, 0, num_to_pad),
            value=0,
        )
        bond_feats_padded = F.pad(
            entry.edge_features,
            pad=(0, 0, 0, num_to_pad, 0, num_to_pad),
            value=0,
        )
        adj_matrix_padded = F.pad(
            entry.adj_matrix,
            pad=(0, num_to_pad, 0, num_to_pad),
            value=0,
        )
        all_targets.append(entry.target)

        all_edge_features.append(bond_feats_padded)
        all_adj_matrices.append(adj_matrix_padded)
        all_node_features.append(node_feats_padded)
    return tuple(
        map(
            torch.stack,
            [
                all_node_features,
                all_edge_features,
                all_adj_matrices,
                all_targets,
            ],
        ),
    )
