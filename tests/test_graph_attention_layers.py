import pytest
import torch
from rdkit import Chem

from graphmodels.datasets import mpnn_dataset
from graphmodels.layers import graph_attention_layers


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
    def test_graph_attention_layer_v3_number_of_layers(
        self,
        n_node_features,
        n_hidden_features,
        num_layers,
    ):
        gat_layer = graph_attention_layers.GraphAttentionLayerV3(
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

    def test_graph_attention_layer_v3_output_shape(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        num_atoms = Chem.MolFromSmiles(smi).GetNumAtoms()

        gat_layer = graph_attention_layers.GraphAttentionLayerV3(
            n_node_features=136,
            n_hidden_features=200,
        )
        out = gat_layer(
            node_features=input_entry.node_features,
            edge_index=input_entry.edge_indices,
        )

        assert out.shape == (num_atoms, 200)

    def test_graph_attention_layer_v3_output(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        mol = Chem.MolFromSmiles(smi)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        gat_layer = graph_attention_layers.GraphAttentionLayerV3(
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
    def test_graph_attention_layer_v2_number_of_layers(
        self,
        n_node_features,
        n_edge_features,
        n_hidden_features,
        num_layers,
    ):
        gat_layer = graph_attention_layers.GraphAttentionLayerV2(
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

    def test_graph_attention_layer_v2_output_shape(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        num_atoms = Chem.MolFromSmiles(smi).GetNumAtoms()

        gat_layer = graph_attention_layers.GraphAttentionLayerV2(
            n_node_features=136,
            n_hidden_features=200,
            n_edge_features=24,
        )
        out = gat_layer(
            node_features=input_entry.node_features,
            edge_index=input_entry.edge_indices,
            edge_features=input_entry.edge_features,
        )

        assert out.shape == (num_atoms, 200)

    def test_graph_attention_layer_v2_output(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        mol = Chem.MolFromSmiles(smi)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        gat_layer = graph_attention_layers.GraphAttentionLayerV2(
            n_node_features=136,
            n_hidden_features=200,
            n_edge_features=24,
            dropout=0.25,
            scaling=0.2,
        )
        att_out = gat_layer.compute_attention(
            node_features=input_entry.node_features,
            edge_index=input_entry.edge_indices,
            edge_features=input_entry.edge_features,
        )

        assert len(att_out) == 3
        assert att_out[0].shape == (num_bonds * 2, 200)
        assert att_out[1].shape == (num_atoms, 200)
        torch.testing.assert_close(att_out[2], input_entry.edge_indices[0])

    @pytest.mark.parametrize(
        "n_node_features,n_hidden_features,num_layers",
        [
            (136, 200, 2),
            (64, 512, 2),
        ],
    )
    def test_graph_attention_layer_v1_number_of_layers(
        self,
        n_node_features,
        n_hidden_features,
        num_layers,
    ):
        gat_layer = graph_attention_layers.GraphAttentionLayerV1(
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

    def test_graph_attention_layer_v1_output_shape(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        num_atoms = Chem.MolFromSmiles(smi).GetNumAtoms()

        gat_layer = graph_attention_layers.GraphAttentionLayerV1(
            n_node_features=136,
            n_hidden_features=200,
        )
        out = gat_layer(
            node_features=input_entry.node_features,
            edge_index=input_entry.edge_indices,
        )

        assert out.shape == (num_atoms, 200)

    def test_graph_attention_layer_v1_output(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        mol = Chem.MolFromSmiles(smi)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        gat_layer = graph_attention_layers.GraphAttentionLayerV1(
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


if __name__ == "_main_":
    pytest.main([__file__])
