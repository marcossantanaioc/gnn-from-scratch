import pytest
import torch
from rdkit import Chem

from neuralfingerprint import constants, datasets


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

    def test_transform(self, smi):
        moldataset = datasets.NeuralFingerprintDataset(
            smiles=(smi,),
            targets=(1.0,),
        )
        atom_features, bond_features, target = moldataset[0]
        assert isinstance(target, float)
        assert isinstance(atom_features, torch.Tensor)
        assert atom_features.shape == (29, constants.NUM_ATOM_FEATURES)
        assert isinstance(bond_features, torch.Tensor)
        assert bond_features.shape == (29, 29, constants.NUM_BOND_FEATURES)


if __name__ == "__main__":
    pytest.main([__file__])
