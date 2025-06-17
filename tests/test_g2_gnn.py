import pytest

from graphmodels.datasets import mpnn_dataset
from graphmodels.layers import g2_gnn, graph_attention_layers


class TestGraphAttentionLayers:
    """Pytests"""

    @pytest.fixture
    def smi(self):
        return "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"

    @pytest.mark.parametrize(
        "concat,n_out_features,num_heads",
        [
            (True, 8, 8),
            (True, 16, 2),
            (False, 8, 8),
            (False, 16, 2),
        ],
    )
    def test_g2_output_shape(self, smi, concat, n_out_features, num_heads):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )
        input_entry = moldataset[0]

        conv_layer = graph_attention_layers.MultiHeadGATLayer(
            n_input_features=input_entry.node_features.size(-1),
            n_out_features=n_out_features,
            num_heads=num_heads,
            concat=concat,
        )

        g2_layer = g2_gnn.G2(conv=conv_layer, p=2)

        out = g2_layer(input_entry.node_features, input_entry.edge_indices)

        if concat:
            assert out.shape == (
                input_entry.node_features.size(0),
                n_out_features * num_heads,
            )
        else:
            assert out.shape == (
                input_entry.node_features.size(0),
                n_out_features,
            )


if __name__ == "_main_":
    pytest.main([__file__])
