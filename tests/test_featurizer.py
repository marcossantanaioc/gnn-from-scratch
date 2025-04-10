import pytest
import torch
from rdkit import Chem

from graphmodels import constants, featurizer


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

    def test_featurize_one_bond(self, molecule):
        bond = molecule.GetBonds()[0]
        features = featurizer._featurize_one_bond(bond)
        assert len(features) == 24
        assert isinstance(features, torch.Tensor)
        assert features.shape == (constants.NUM_BOND_FEATURES,)

    def test_featurize_bonds(self, molecule):
        num_atoms = molecule.GetNumAtoms()
        features = featurizer.featurize_bonds(molecule)
        assert len(features) == num_atoms
        assert isinstance(features, torch.Tensor)
        assert features.shape == (
            num_atoms,
            num_atoms,
            constants.NUM_BOND_FEATURES,
        )


if __name__ == "__main__":
    pytest.main([__file__])
