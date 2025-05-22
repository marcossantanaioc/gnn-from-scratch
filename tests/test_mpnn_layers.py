import pytest
import torch
from rdkit import Chem
from torch import nn

from graphmodels.datasets import mpnn_dataset
from graphmodels.layers import mpnn_layers


class TestLayers:
    """Pytests"""

    @pytest.fixture
    def smi(self):
        return "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"

    @pytest.fixture
    def molecule(self, smi):
        return Chem.MolFromSmiles(smi)

    @pytest.mark.parametrize(
        "n_edge_features, n_node_features, n_towers",
        [
            (24, 136, 8),
            (50, 64, 4),
        ],
    )
    def test_multitower_edge_layer_shape(
        self,
        n_edge_features,
        n_node_features,
        n_towers,
    ):
        edge_network = mpnn_layers.MultiTowerEdge(
            n_edge_features=n_edge_features,
            n_node_features=n_node_features,
            n_towers=n_towers,
        )

        tower_dim = n_node_features // n_towers
        assert edge_network.edgetower[0].weight.shape == (
            tower_dim * n_node_features,
            n_edge_features,
        )

    def test_multitower_edge_layer_output_shape(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        num_bonds = Chem.MolFromSmiles(smi).GetNumBonds() * 2

        edge_network = mpnn_layers.MultiTowerEdge(
            n_edge_features=24,
            n_towers=8,
            n_node_features=136,
        )
        message = edge_network(
            edge_features=input_entry.edge_features,
            node_features=input_entry.node_features,
            edge_index=input_entry.edge_indices,
        )

        assert message.shape == (num_bonds, 136)

    @pytest.mark.parametrize(
        "e_feats,e_hidden,n_node_features, n_update_steps,layers",
        [
            (24, 100, 136, 2, 3),
            (50, 512, 20, 3, 4),
        ],
    )
    def test_edge_layer_number_of_layers(
        self,
        e_feats,
        e_hidden,
        n_node_features,
        n_update_steps,
        layers,
    ):
        edge_network = mpnn_layers.EdgeLayer(
            n_edge_features=e_feats,
            n_edge_hidden_features=e_hidden,
            n_node_features=n_node_features,
            n_update_steps=n_update_steps,
        )
        assert (
            len(
                [
                    lay
                    for lay in edge_network.edgelayer.modules()
                    if isinstance(lay, torch.nn.Linear)
                ],
            )
            == layers
        )  # Includes input layer

    def test_edge_layer_output_shape(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        num_bonds = Chem.MolFromSmiles(smi).GetNumBonds() * 2

        edge_network = mpnn_layers.EdgeLayer(
            n_edge_features=24,
            n_edge_hidden_features=200,
            n_node_features=136,
            n_update_steps=2,
        )
        message = edge_network(
            edge_features=input_entry.edge_features,
            node_features=input_entry.node_features,
            edge_index=input_entry.edge_indices,
        )

        assert message.shape == (num_bonds, 136)

    @pytest.mark.parametrize(
        "n_node_features,n_edge_features,n_update_steps",
        [
            (136, 24, 2),
            (200, 16, 3),
        ],
    )
    def test_update_gru_cell_weights_and_updates(
        self,
        n_node_features,
        n_edge_features,
        n_update_steps,
    ):
        update_network = mpnn_layers.MessagePassingLayer(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_update_steps=n_update_steps,
        )

        assert isinstance(update_network.update_cell, nn.GRUCell)
        assert update_network.n_update_steps == n_update_steps
        assert update_network.update_cell.weight_hh.size(1) == n_node_features
        assert update_network
        assert update_network.update_cell.weight_ih.size(1) == n_node_features

    def test_update_layer_output_shape(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]

        update_network = mpnn_layers.MessagePassingLayer(
            n_node_features=136,
            n_edge_features=24,
            n_update_steps=3,
        )

        out = update_network(
            node_features=input_entry.node_features,
            edge_features=input_entry.edge_features,
            edge_index=input_entry.edge_indices,
        )

        assert out.shape == input_entry.node_features.shape

    @pytest.mark.parametrize(
        "n_node_features, n_hidden_features, n_out_features, num_layers",
        [
            (24, 200, 136, 2),
            (50, 512, 20, 3),
        ],
    )
    def test_readout_layer_number_of_layers(
        self,
        n_node_features,
        n_hidden_features,
        n_out_features,
        num_layers,
    ):
        readout_network = mpnn_layers.ReadoutLayer(
            n_node_features=n_node_features,
            n_hidden_features=n_hidden_features,
            n_out_features=n_out_features,
            num_layers=num_layers,
        )
        assert (
            len(
                [
                    lay
                    for lay in readout_network.readout.modules()
                    if isinstance(lay, torch.nn.Linear)
                ],
            )
            == num_layers
        )

    def test_readout_layer_output_shape(self, smi):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]

        readout_network = mpnn_layers.ReadoutLayer(
            n_node_features=136,
            n_hidden_features=512,
            n_out_features=1,
            num_layers=3,
        )
        batch_vector = torch.zeros(input_entry.node_features.size(0)).to(
            torch.int32,
        )

        out = readout_network(
            node_features=input_entry.node_features,
            batch_vector=batch_vector,
        )

        assert out.shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__])
