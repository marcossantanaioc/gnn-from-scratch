import torch
from torch.nn import F

from graphmodels import datasets


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
