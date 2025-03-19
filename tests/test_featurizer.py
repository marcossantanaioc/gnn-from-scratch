from rdkit import Chem
import pytest


class TestMolFeaturizer:
    """
    Pytests
    """

    @pytest.fixture
    def smi(self):
        return "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"
    
    @pytest.fixture
    def molecule(self):
        return Chem.MolFromSmiles(smi)
    
    def test_featurize_atom(self, molecule):
        assert isinstance(molecule) == Chem.Mol

if __name__ == "__main__":
    pytest.main([__file__])