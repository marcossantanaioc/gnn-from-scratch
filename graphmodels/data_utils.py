import torch
from torch.nn import functional as F  # noqa: N812
import dataclasses
from graphmodels import datasets


@dataclasses.dataclass(kw_only=True, frozen=True)
class NeuralGraphFingerprintBatch:
    """Store batch information for the neural graph fingerprint model."""

    adj_matrix: torch.Tensor
    edge_index: torch.Tensor
    atom_features: torch.Tensor
    targets: torch.Tensor

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass(kw_only=True, frozen=True)
class MPNNBatch:
    """Store batch information for the MPNNv1 model."""

    # adj_matrix: torch.Tensor
    edge_index: torch.Tensor
    batch_vector: torch.Tensor
    atom_features: torch.Tensor
    bond_features: torch.Tensor
    targets: torch.Tensor

    def to_dict(self):
        return dataclasses.asdict(self)


def mpnn_collate_diag(batch):
    """Generates a batch of inputs for MPNN."""
    # Build batch vector
    num_atoms_per_mol = [nf.atom_features.size(0) for nf in batch]
    batch_vector = torch.cat(
        [
            torch.full((n,), i, dtype=torch.long)
            for i, n in enumerate(num_atoms_per_mol)
        ]
    )  # [total_atoms], values in [0, batch_size-1]

    all_edge_indices = torch.block_diag(*[x.edge_indices for x in batch])
    all_atom_features = torch.concat([x.atom_features for x in batch], dim=0)
    all_bond_features = torch.concat([x.bond_features for x in batch], dim=0)
    targets = torch.stack([x.target for x in batch])

    return MPNNBatch(
        batch_vector=batch_vector,
        edge_index=all_edge_indices,
        atom_features=all_atom_features,
        bond_features=all_bond_features,
        targets=targets,
    )


def neuralgraph_collate_diag(batch):
    """Generates a batch of inputs for NeuralGraphFingerprint."""
    # Build batch vector
    num_atoms_per_mol = [nf.atom_features.size(0) for nf in batch]
    batch_vector = torch.cat(
        [
            torch.full((n,), i, dtype=torch.long)
            for i, n in enumerate(num_atoms_per_mol)
        ]
    )  # [total_atoms], values in [0, batch_size-1]

    all_adj_matrix = torch.block_diag(*[x.adj_matrix for x in batch])
    all_atom_features = torch.concat([x.atom_features for x in batch], dim=0)
    targets = torch.stack([x.target for x in batch])

    return NeuralGraphFingerprintBatch(
        adj_matrix=all_adj_matrix,
        edge_index=batch_vector,
        atom_features=all_atom_features,
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
    all_atom_features = []
    all_bond_features = []
    all_adj_matrices = []
    all_targets = []

    # Get max number of atoms in data
    for entry in batch:
        num_to_pad = max_num_atoms - entry.atom_features.shape[0]
        atom_feats_padded = F.pad(
            entry.atom_features,
            pad=(0, 0, 0, num_to_pad),
            value=0,
        )
        bond_feats_padded = F.pad(
            entry.bond_features,
            pad=(0, 0, 0, num_to_pad, 0, num_to_pad),
            value=0,
        )
        adj_matrix_padded = F.pad(
            entry.adj_matrix,
            pad=(0, num_to_pad, 0, num_to_pad),
            value=0,
        )
        all_targets.append(entry.target)

        all_bond_features.append(bond_feats_padded)
        all_adj_matrices.append(adj_matrix_padded)
        all_atom_features.append(atom_feats_padded)
    return tuple(
        map(
            torch.stack,
            [
                all_atom_features,
                all_bond_features,
                all_adj_matrices,
                all_targets,
            ],
        ),
    )
