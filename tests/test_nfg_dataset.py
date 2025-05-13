import pytest
import torch
from rdkit import Chem

from graphmodels import constants
from graphmodels.datasets import ngf_dataset


class TestNeuralFingerprintDataset:
    """
    Pytests
    """

    @pytest.fixture
    def smi(self):
        return "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"

    @pytest.fixture
    def molecule(self, smi):
        return Chem.MolFromSmiles(smi)

    def test_dataset_len(self, smi):
        moldataset = ngf_dataset.NeuralFingerprintDataset(
            smiles=(smi,),
            targets=(1.0,),
        )
        assert len(moldataset) == 1

    def test_fetch_one_from_dataset(self, smi):
        moldataset = ngf_dataset.NeuralFingerprintDataset(
            smiles=(smi,),
            targets=(1.0,),
        )
        try:
            moldataset[0]
        except IndexError:
            pytest.fail("The dataset has zero entries.")

    def test_fetch_features_from_dataset(self, smi):
        moldataset = ngf_dataset.NeuralFingerprintDataset(
            smiles=(smi,),
            targets=(1.0,),
        )
        input_entry = moldataset[0]
        assert isinstance(input_entry, ngf_dataset.NeuralFingerprintEntry)
        assert isinstance(input_entry.target, torch.Tensor)
        assert isinstance(input_entry.node_features, torch.Tensor)
        assert input_entry.node_features.shape == (
            29,
            constants.NUM_NODE_FEATURES,
        )
        assert isinstance(input_entry.edge_features, torch.Tensor)
        assert input_entry.edge_features.shape == (
            29,
            29,
            constants.NUM_EDGE_FEATURES,
        )
        assert input_entry.adj_matrix.shape == (29, 29)


if __name__ == "__main__":
    pytest.main([__file__])
