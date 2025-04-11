import pytest
import torch
from rdkit import Chem

from graphmodels import constants, datasets


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
        moldataset = datasets.NeuralFingerprintDataset(
            smiles=(smi,),
            targets=(1.0,),
        )
        assert len(moldataset) == 1

    def test_fetch_one_from_dataset(self, smi):
        moldataset = datasets.NeuralFingerprintDataset(
            smiles=(smi,),
            targets=(1.0,),
        )
        try:
            moldataset[0]
        except IndexError:
            pytest.fail("The dataset has zero entries.")

    def test_fetch_features_from_dataset(self, smi):
        moldataset = datasets.NeuralFingerprintDataset(
            smiles=(smi,),
            targets=(1.0,),
        )
        input_entry = moldataset[0]
        assert isinstance(input_entry, datasets.NeuralFingerprintEntry)
        assert isinstance(input_entry.target, torch.Tensor)
        assert isinstance(input_entry.atom_features, torch.Tensor)
        assert input_entry.atom_features.shape == (
            29,
            constants.NUM_ATOM_FEATURES,
        )
        assert isinstance(input_entry.bond_features, torch.Tensor)
        assert input_entry.bond_features.shape == (
            29,
            29,
            constants.NUM_BOND_FEATURES,
        )
        assert input_entry.adj_matrix.shape == (29, 29)


if __name__ == "__main__":
    pytest.main([__file__])
