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
        assert features.shape == (29, constants.NUM_NODE_FEATURES)

    def test_featurize_one_bond(self, molecule):
        bond = molecule.GetBonds()[0]
        features = featurizer._featurize_one_bond(bond)
        assert len(features) == 24
        assert isinstance(features, torch.Tensor)
        assert features.shape == (constants.NUM_EDGE_FEATURES,)

    @pytest.mark.parametrize(
        "input_smiles, num_atoms",
        [
            (
                "C",
                1,
            ),
            ("O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl", 29),
        ],
    )
    def test_featurize_bonds_per_atom(
        self,
        input_smiles,
        num_atoms,
    ):
        mol = Chem.MolFromSmiles(input_smiles)
        features = featurizer.featurize_bonds_per_atom(mol)
        assert len(features) == num_atoms
        assert isinstance(features, torch.Tensor)
        assert features.shape == (
            num_atoms,
            num_atoms,
            constants.NUM_EDGE_FEATURES,
        )
        # Test no bond case.
        if num_atoms == 1:
            expected_t = torch.zeros(
                (
                    num_atoms,
                    num_atoms,
                    constants.NUM_EDGE_FEATURES,
                ),
            )
            torch.testing.assert_close(
                features,
                expected_t,
            )

    def test_featurize_molecule_with_no_atoms(self):
        invalid_mol = Chem.MolFromSmiles("")
        with pytest.raises(featurizer.NoAtomError):
            featurizer.featurize_atoms(invalid_mol)

    @pytest.mark.parametrize(
        "input_smiles, expected_size",
        [
            (
                "C",
                1,
            ),
            ("O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl", 32),
        ],
    )
    def test_featurize_bonds(self, input_smiles, expected_size):
        mol = Chem.MolFromSmiles(input_smiles)
        features = featurizer.featurize_bonds(mol)
        assert len(features) == expected_size
        assert isinstance(features, torch.Tensor)
        assert features.shape == (
            expected_size,
            constants.NUM_EDGE_FEATURES,
        )


if __name__ == "__main__":
    pytest.main([__file__])
