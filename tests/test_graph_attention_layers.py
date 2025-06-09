from typing import NamedTuple

import pytest
import torch
from rdkit import Chem

from graphmodels.datasets import mpnn_dataset
from graphmodels.layers import graph_attention_layers


class TestDataset(NamedTuple):
    __test__ = False
    dset: mpnn_dataset.MPNNEntry
    cat_node_features: torch.Tensor
    cat_edge_features: torch.Tensor
    num_atoms: int
    num_bonds: int


def get_embedding_input(
    smi: str,
):
    mol = Chem.MolFromSmiles(smi)
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds() * 2
    moldataset = mpnn_dataset.MPNNDataset(
        smiles=(smi,),
        targets=(1.0,),
    )

    input_entry = moldataset[0]

    edge_features = torch.zeros(num_bonds)
    bond_idx = 0
    for atom in mol.GetAtoms():
        bonds = atom.GetBonds()
        for bond in bonds:
            edge_features[bond_idx] = int(bond.GetBondTypeAsDouble())
            bond_idx += 1

    input_entry = moldataset[0]
    node_features = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
    ).to(torch.int32)
    edge_features = edge_features.to(torch.int32)

    return TestDataset(
        dset=input_entry,
        cat_node_features=node_features,
        cat_edge_features=edge_features,
        num_atoms=num_atoms,
        num_bonds=num_bonds,
    )


class TestGraphAttentionLayers:
    """Pytests"""

    @pytest.fixture
    def smi(self):
        return "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"

    @pytest.mark.parametrize(
        "n_node_features,n_hidden_features,num_layers",
        [
            (136, 200, 2),
            (64, 512, 2),
        ],
    )
    def test_graph_attention_layer_skip_number_of_layers(
        self,
        n_node_features,
        n_hidden_features,
        num_layers,
    ):
        gat_layer = graph_attention_layers.GraphAttentionLayerSkip(
            n_node_features=n_node_features,
            n_hidden_features=n_hidden_features,
        )
        assert (
            len(
                [
                    layer
                    for layer in gat_layer.modules()
                    if isinstance(layer, torch.nn.Linear)
                ],
            )
            == num_layers
        )

    def test_graph_attention_layer_skip_output_shape(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        num_atoms = Chem.MolFromSmiles(smi).GetNumAtoms()

        gat_layer = graph_attention_layers.GraphAttentionLayerSkip(
            n_node_features=136,
            n_hidden_features=200,
        )
        out = gat_layer(
            node_features=input_entry.node_features,
            edge_index=input_entry.edge_indices,
        )

        assert out.shape == (num_atoms, 200)

    def test_graph_attention_layer_skip_output(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        mol = Chem.MolFromSmiles(smi)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        gat_layer = graph_attention_layers.GraphAttentionLayerSkip(
            n_node_features=136,
            n_hidden_features=200,
            dropout=0.25,
            scaling=0.2,
        )
        att_out = gat_layer.compute_attention(
            node_features=input_entry.node_features,
            edge_index=input_entry.edge_indices,
        )

        assert len(att_out) == 3
        assert att_out[0].shape == (num_bonds * 2, 200)
        assert att_out[1].shape == (num_atoms, 200)
        torch.testing.assert_close(att_out[2], input_entry.edge_indices[0])

    @pytest.mark.parametrize(
        "n_node_features,n_edge_features,n_hidden_features,num_layers",
        [
            (136, 24, 200, 3),
            (64, 24, 512, 3),
        ],
    )
    def test_graph_attention_layer_edge_number_of_layers(
        self,
        n_node_features,
        n_edge_features,
        n_hidden_features,
        num_layers,
    ):
        gat_layer = graph_attention_layers.GraphAttentionLayerEdge(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_hidden_features=n_hidden_features,
        )
        assert (
            len(
                [
                    layer
                    for layer in gat_layer.modules()
                    if isinstance(layer, torch.nn.Linear)
                ],
            )
            == num_layers
        )

    def test_graph_attention_layer_edge_output_shape(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )
        mol = Chem.MolFromSmiles(smi)
        input_entry = moldataset[0]
        num_atoms = mol.GetNumAtoms()

        gat_layer = graph_attention_layers.GraphAttentionLayerEdge(
            n_node_features=136,
            n_hidden_features=200,
            n_edge_features=24,
        )
        out, edge_out = gat_layer(
            node_features=input_entry.node_features,
            edge_index=input_entry.edge_indices,
            edge_features=input_entry.edge_features,
        )

        assert out.shape == (num_atoms, 200)
        assert edge_out.shape == (input_entry.edge_indices.shape[1], 200)

    def test_graph_attention_layer_edge_output(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        mol = Chem.MolFromSmiles(smi)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        gat_layer = graph_attention_layers.GraphAttentionLayerEdge(
            n_node_features=136,
            n_hidden_features=200,
            n_edge_features=24,
            dropout=0.25,
            scaling=0.2,
        )
        message, h, edge_h, target_nodes = gat_layer.compute_attention(
            node_features=input_entry.node_features,
            edge_index=input_entry.edge_indices,
            edge_features=input_entry.edge_features,
        )

        assert message.shape == (num_bonds * 2, 200)
        assert h.shape == (num_atoms, 200)
        assert edge_h.shape == (num_bonds * 2, 200)
        torch.testing.assert_close(target_nodes, input_entry.edge_indices[0])

    @pytest.mark.parametrize(
        "n_node_features,n_edge_features,n_hidden_features,num_heads",
        [
            (136, 24, 200, 8),
            (64, 45, 512, 4),
        ],
    )
    def test_multihead_graph_edge_attention_layer(
        self,
        n_node_features,
        n_edge_features,
        n_hidden_features,
        num_heads,
    ):
        gat_layer = graph_attention_layers.MultiHeadEdgeGATLayer(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_hidden_features=n_hidden_features,
            dropout=0.1,
            num_heads=num_heads,
        )
        assert gat_layer.attn.weight.shape == torch.Size(
            [1, 3 * n_hidden_features],
        )
        assert gat_layer.w.weight.shape == torch.Size(
            [n_hidden_features * num_heads, n_node_features],
        )

    @pytest.mark.parametrize(
        "num_heads,n_hidden_features,concat",
        [
            (8, 8, True),
            (4, 8, False),
            (1, 8, True),
            (2, 24, False),
            (3, 17, True),
        ],
    )
    def test_multihead_graph_edge_attention_layer_output_shape(
        self,
        smi,
        num_heads,
        n_hidden_features,
        concat,
    ):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]

        gat_layer = graph_attention_layers.MultiHeadEdgeGATLayer(
            n_node_features=136,
            n_edge_features=24,
            n_hidden_features=n_hidden_features,
            num_heads=num_heads,
            concat=concat,
        )
        out_n, out_e = gat_layer(
            node_features=input_entry.node_features,
            edge_features=input_entry.edge_features,
            edge_index=input_entry.edge_indices,
        )

        if concat:
            assert out_n.shape == (
                input_entry.node_features.size(0),
                num_heads * n_hidden_features,
            )
        else:
            assert out_n.shape == (
                input_entry.node_features.size(0),
                n_hidden_features,
            )

    @pytest.mark.parametrize(
        "n_node_features,n_hidden_features,num_heads",
        [
            (136, 200, 8),
            (64, 512, 4),
        ],
    )
    def test_multihead_graph_attention_layer_v2(
        self,
        n_node_features,
        n_hidden_features,
        num_heads,
    ):
        gat_layer = graph_attention_layers.MultiHeadGATLayer(
            n_node_features=n_node_features,
            n_hidden_features=n_hidden_features,
            num_heads=num_heads,
        )
        assert gat_layer.attn.shape == torch.Size(
            [1, num_heads, 2 * n_hidden_features],
        )
        assert gat_layer.w.weight.shape == torch.Size(
            [n_hidden_features * num_heads, n_node_features],
        )

    @pytest.mark.parametrize(
        "num_heads,n_hidden_features,concat",
        [
            (8, 8, True),
            (4, 8, False),
            (1, 8, True),
            (2, 24, False),
            (3, 17, True),
        ],
    )
    def test_multihead_graph_attention_layer_v2_output_shape(
        self,
        smi,
        num_heads,
        n_hidden_features,
        concat,
    ):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        num_atoms = Chem.MolFromSmiles(smi).GetNumAtoms()

        gat_layer = graph_attention_layers.MultiHeadGATLayer(
            n_node_features=136,
            n_hidden_features=n_hidden_features,
            num_heads=num_heads,
            concat=concat,
        )
        out = gat_layer(
            node_features=input_entry.node_features,
            edge_index=input_entry.edge_indices,
        )

        if concat:
            assert out.shape == (num_atoms, num_heads * n_hidden_features)
        else:
            assert out.shape == (num_atoms, n_hidden_features)

    @pytest.mark.parametrize(
        "n_node_dict,n_edge_dict,embedding_dim,n_hidden_features,num_heads",
        [
            (65, 5, 64, 8, 8),
            (20, 6, 24, 4, 2),
        ],
    )
    def test_embedding_gat_layer(
        self,
        n_node_dict,
        n_edge_dict,
        embedding_dim,
        n_hidden_features,
        num_heads,
    ):
        gat_layer = graph_attention_layers.EmbeddingGATEdge(
            n_node_dict=n_node_dict,
            n_edge_dict=n_edge_dict,
            embedding_dim=embedding_dim,
            n_hidden_features=n_hidden_features,
            num_heads=num_heads,
        )
        assert gat_layer.attn.weight.shape == torch.Size(
            [1, 3 * n_hidden_features],
        )
        assert gat_layer.w.weight.shape == torch.Size(
            [n_hidden_features * num_heads, embedding_dim],
        )

    @pytest.mark.parametrize(
        "embedding_dim,n_hidden_features,num_heads,concat",
        [
            (8, 8, 8, True),
            (16, 4, 8, True),
            (32, 1, 8, True),
            (64, 2, 24, True),
            (128, 3, 17, True),
        ],
    )
    def test_embedding_gat_layer_output_shape(
        self,
        smi,
        embedding_dim,
        n_hidden_features,
        num_heads,
        concat,
    ):
        test_entry = get_embedding_input(smi)

        gat_layer = graph_attention_layers.EmbeddingGATEdge(
            n_node_dict=65,
            n_edge_dict=6,
            embedding_dim=embedding_dim,
            n_hidden_features=n_hidden_features,
            num_heads=num_heads,
            concat=concat,
            batch_norm=False,
        )
        out, _ = gat_layer(
            node_features=test_entry.cat_node_features,
            edge_features=test_entry.cat_edge_features,
            edge_index=test_entry.dset.edge_indices,
        )

        if concat:
            assert out.shape == (
                test_entry.cat_node_features.size(0),
                num_heads * n_hidden_features,
            )
        else:
            assert out.shape == (
                test_entry.cat_node_features.size(0),
                n_hidden_features,
            )


if __name__ == "_main_":
    pytest.main([__file__])
