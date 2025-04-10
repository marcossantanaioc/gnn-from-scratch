import pytest
import torch
from rdkit import Chem

from neuralfingerprint import constants, featurizer


class TestMolFeaturizer:
    """
    Pytests
    """

    @pytest.fixture
    def smi(self):
        return "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"

    @pytest.fixture
    def molecule(self, smi):
        return Chem.MolFromSmiles(smi)

    def test_featurize_atoms(self, molecule):
        num_atoms = molecule.GetNumAtoms()
        features = featurizer.featurize_atoms(molecule)
        assert len(features) == num_atoms
        assert isinstance(features, torch.Tensor)
        assert features.shape == (29, constants.NUM_ATOM_FEATURES)

    def test_featurize_bonds(self, molecule):
        num_atoms = molecule.GetNumBonds()
        features = featurizer.featurize_bonds(molecule)
        assert len(features) == num_atoms
        assert isinstance(features, torch.Tensor)
        assert features.shape == (32, constants.NUM_BOND_FEATURES)


if __name__ == "__main__":
    pytest.main([__file__])
