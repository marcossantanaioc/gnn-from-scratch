from graphmodels import datasets, layers
import pytest
from rdkit import Chem
import torch


class TestLayers:
    """Pytests"""

    @pytest.fixture
    def smi(self):
        return "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"

    @pytest.fixture
    def molecule(self, smi):
        return Chem.MolFromSmiles(smi)

    @pytest.mark.parametrize(
        "n_edge_features, n_hidden_features, n_node_features, passes,"
        " expected_num_layers",
        [
            (24, 200, 136, 2, 3),
            (50, 512, 20, 3, 4),
        ],
    )
    def test_edge_layer_number_of_layers(
        self,
        n_edge_features,
        n_hidden_features,
        n_node_features,
        passes,
        expected_num_layers,
    ):
        edge_network = layers.EdgeLayer(
            n_edge_features=n_edge_features,
            n_hidden_features=n_hidden_features,
            n_node_features=n_node_features,
            passes=passes,
        )
        assert (
            len(
                [
                    l
                    for l in edge_network.edgelayer.modules()
                    if isinstance(l, torch.nn.Linear)
                ]
            )
            == expected_num_layers
        )  # Includes input layer

    def test_edge_layer_output_shape(self, smi):
        moldataset = datasets.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        num_bonds = Chem.MolFromSmiles(smi).GetNumBonds()

        edge_network = layers.EdgeLayer(
            n_edge_features=24,
            n_hidden_features=200,
            n_node_features=136,
            passes=2,
        )
        message = edge_network(
            (
                input_entry.bond_features,
                input_entry.atom_features,
                input_entry.edge_indices,
            )
        )

        assert message.shape == (num_bonds * 2, 136)

    @pytest.mark.parametrize(
        "n_node_features, n_hidden_features, num_layers",
        [
            (136, 512, 2),
            (200, 100, 3),
        ],
    )
    def test_update_layer_number_of_layers(
        self,
        n_node_features,
        n_hidden_features,
        num_layers,
    ):
        update_network = layers.UpdateLayer(
            n_node_features=n_node_features,
            n_hidden_features=n_hidden_features,
            num_layers=num_layers,
        )
        assert (
            max(
                [
                    l.num_layers
                    for l in update_network.update_layers.modules()
                    if isinstance(l, torch.nn.GRU)
                ]
            )
            == num_layers
        )

    def test_update_layer_output_shape(self, smi):
        moldataset = datasets.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        num_bonds = Chem.MolFromSmiles(smi).GetNumBonds()

        update_network = layers.UpdateLayer(
            n_node_features=136,
            n_hidden_features=512,
            num_layers=3,
        )
        message = torch.rand(32, 136)

        out = update_network(
            (message, input_entry.atom_features, input_entry.edge_indices)
        )
        assert out.shape == input_entry.atom_features.shape

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
        readout_network = layers.ReadoutLayer(
            n_node_features=n_node_features,
            n_hidden_features=n_hidden_features,
            n_out_features=n_out_features,
            num_layers=num_layers,
        )
        assert (
            len(
                [
                    l
                    for l in readout_network.readout.modules()
                    if isinstance(l, torch.nn.Linear)
                ]
            )
            == num_layers
        )

    def test_readout_layer_output_shape(self, smi):
        moldataset = datasets.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]

        readout_network = layers.ReadoutLayer(
            n_node_features=136,
            n_hidden_features=512,
            n_out_features=1,
            num_layers=3,
        )
        batch_vector = torch.zeros(input_entry.atom_features.size(0)).to(
            torch.int32
        )

        out = readout_network((input_entry.atom_features, batch_vector))

        assert out.shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__])
