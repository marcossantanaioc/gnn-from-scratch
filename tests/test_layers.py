from graphmodels import datasets, layers
import pytest
from rdkit import Chem
import torch

class TestEdgeLayer:
    """Pytests"""
    
    @pytest.fixture
    def smi(self):
        return "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"

    @pytest.fixture
    def molecule(self, smi):
        return Chem.MolFromSmiles(smi)

    @pytest.mark.parametrize(
      "n_input_features, n_hidden_features, n_node_features, passes,"
      " expected_num_layers",
      [
          (24, 200, 136, 2, 3),
          (50, 512, 20, 3, 4),
      ],
    )
    def test_edge_layer_number_of_layers(self,
                        n_input_features,
                        n_hidden_features,
                        n_node_features,
                        passes,
                        expected_num_layers):
        
        edge_network = layers.EdgeLayer(
        n_input_features=n_input_features,
        n_hidden_features=n_hidden_features,
        n_node_features=n_node_features,
        passes=passes,
    )
        assert (
        len([
            l
            for l in edge_network.edgelayer.modules()
            if isinstance(l, torch.nn.Linear)
        ])
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
        n_input_features=24,
        n_hidden_features=200,
        n_node_features=136,
        passes=2,
    )
        message = edge_network((input_entry.bond_features,
                                input_entry.atom_features,
                                input_entry.edge_indices))
        
        assert message.shape == (num_bonds, 136)


if __name__ == "__main__":
    pytest.main([__file__])
