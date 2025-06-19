import pytest
import torch
from torch import nn

from graphmodels.datasets import mpnn_dataset
from graphmodels.layers import gin


class TestGINLayer:
    """Pytests"""

    @pytest.fixture
    def smi(self):
        return "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"

    @pytest.mark.parametrize(
        "n_input_features,n_hidden_features",
        [
            (136, 200),
            (64, 512),
        ],
    )
    def test_gin_layer(
        self,
        n_input_features,
        n_hidden_features,
    ):
        gin_layer = gin.GINLayer(
            n_input_features=n_input_features,
            n_hidden_features=n_hidden_features,
        )

        assert len(gin_layer.mlp) == 4
        assert (
            len(
                [
                    layer
                    for layer in gin_layer.mlp
                    if isinstance(layer, nn.Linear)
                ],
            )
            == 2
        )
        assert (
            len([act for act in gin_layer.mlp if isinstance(act, nn.ReLU)])
            == 2
        )
        assert gin_layer.eta.shape == torch.Size([1, 1])

    @pytest.mark.parametrize("n_hidden_features", [2, 4, 8, 7, 13, 128])
    def test_gin_layer_output_shape(
        self,
        smi,
        n_hidden_features,
    ):
        moldataset = mpnn_dataset.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]

        gin_layer = gin.GINLayer(
            n_input_features=input_entry.node_features.size(-1),
            n_hidden_features=n_hidden_features,
        )
        out = gin_layer(input_entry.node_features, input_entry.edge_indices)

        assert out.shape == (
            input_entry.node_features.size(0),
            n_hidden_features,
        )


if __name__ == "_main_":
    pytest.main([__file__])
