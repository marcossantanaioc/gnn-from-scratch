from graphmodels.datasets import mpnn_dataset
from graphmodels.layers import graph_attention_layers
import pytest
from rdkit import Chem
import torch


class TestLayers:
    """Pytests"""

    @pytest.fixture
    def smi(self):
        return "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"

    @pytest.mark.parametrize(
        "n_node_features, n_hidden_features, expected_num_layers",
        [
            (136, 200, 2),
            (64, 512, 2),
        ],
    )
    def test_graph_attention_layer_number_of_layers(
        self,
        n_node_features,
        n_hidden_features,
        expected_num_layers,
    ):
        gat_layer = graph_attention_layers.SimpleGAT(
            n_node_features=n_node_features,
            n_hidden_features=n_hidden_features,
        )
        assert (
            len(
                [
                    layer
                    for layer in gat_layer.modules()
                    if isinstance(layer, torch.nn.Linear)
                ]
            )
            == expected_num_layers
        )  # Includes input layer

    def test_graph_attention_layer_output_shape(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        num_atoms = Chem.MolFromSmiles(smi).GetNumAtoms()

        gat_layer = graph_attention_layers.SimpleGAT(
            n_node_features=136,
            n_hidden_features=200,
        )
        out = gat_layer(
            input_entry.node_features,
            input_entry.edge_indices,
        )

        assert out.shape == (num_atoms, 200)

    def test_graph_attention_attention_output(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        mol = Chem.MolFromSmiles(smi)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        gat_layer = graph_attention_layers.SimpleGAT(
            n_node_features=136,
            n_hidden_features=200,
        )
        att_out = gat_layer.compute_attention(
            input_entry.node_features, input_entry.edge_indices
        )

        assert len(att_out) == 3
        # Check messages have the right shape
        assert att_out[0].shape == (num_bonds, 200)
        # Check if node features have correct shape
        assert att_out[1].shape == (num_atoms, 200)
        # Check if edge indices of target nodes are correct
        torch.testing.assert_close(att_out[2], input_entry.edge_indices[0])


if __name__ == "__main__":
    pytest.main([__file__])
