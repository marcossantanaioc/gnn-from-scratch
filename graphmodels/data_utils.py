import torch
from torch.nn import functional as F  # noqa: N812
import dataclasses
from collections.abc import Sequence
from graphmodels import datasets


@dataclasses.dataclass(kw_only=True, frozen=True)
class NeuralGraphFingerprintBatch:
    """Store batch information for the neural graph fingerprint model."""

    adj_matrix: torch.Tensor
    edge_index: torch.Tensor
    node_features: torch.Tensor
    targets: torch.Tensor

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass(kw_only=True, frozen=True)
class MPNNBatch:
    """Store batch information for the MPNNv1 model."""

    edge_index: torch.Tensor
    batch_vector: torch.Tensor
    node_features: torch.Tensor
    edge_features: torch.Tensor
    targets: torch.Tensor

    def to_dict(self):
        return dataclasses.asdict(self)


def create_batch_edge_index(edge_indices: Sequence[torch.Tensor]):
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
    for idx, edge_index in enumerate(edge_indices):
        num_nodes = edge_index.max() + 1

        edge_index_new = edge_index + offset

        to_concat.append(edge_index_new)
        offset += num_nodes
        total_num_nodes += num_nodes

    return torch.cat(to_concat, dim=-1)


def mpnn_collate_diag(batch):
    """Generates a batch of inputs for MPNN."""
    # Build batch vector
    num_atoms_per_mol = [nf.node_features.size(0) for nf in batch]
    batch_vector = torch.cat(
        [
            torch.full((n,), i, dtype=torch.long)
            for i, n in enumerate(num_atoms_per_mol)
        ]
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


def neuralgraph_collate_diag(batch):
    """Generates a batch of inputs for NeuralGraphFingerprint."""
    # Build batch vector
    num_atoms_per_mol = [nf.node_features.size(0) for nf in batch]
    batch_vector = torch.cat(
        [
            torch.full((n,), i, dtype=torch.long)
            for i, n in enumerate(num_atoms_per_mol)
        ]
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


def neuralgraph_longest_collate(
    batch: list[datasets.NeuralFingerprintEntry],
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
