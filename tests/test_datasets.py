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


class TestMPNNDataset:
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
        moldataset = datasets.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )
        assert len(moldataset) == 1

    def test_fetch_one_from_dataset(self, smi):
        moldataset = datasets.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )
        try:
            moldataset[0]
        except IndexError:
            pytest.fail("The dataset has zero entries.")

    def test_fetch_features_from_dataset(self, smi):
        moldataset = datasets.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
        )

        input_entry = moldataset[0]
        num_bonds = Chem.MolFromSmiles(smi).GetNumBonds()

        assert isinstance(input_entry, datasets.MPNNEntry)
        assert isinstance(input_entry.target, torch.Tensor)
        assert isinstance(input_entry.node_features, torch.Tensor)
        assert input_entry.node_features.shape == (
            29,
            constants.NUM_NODE_FEATURES,
        )
        assert isinstance(input_entry.edge_features, torch.Tensor)
        assert input_entry.edge_features.shape == (
            num_bonds,
            constants.NUM_EDGE_FEATURES,
        )
        assert input_entry.adj_matrix.shape == (29, 29)
        assert input_entry.edge_indices.shape == (2, num_bonds * 2)

    def test_fetch_features_from_dataset_with_master_node(self, smi):
        moldataset = datasets.MPNNDataset(
            smiles=(smi,),
            targets=(1.0,),
            add_master_node=True,
        )

        input_entry = moldataset[0]
        num_bonds = Chem.MolFromSmiles(smi).GetNumBonds()
        num_nodes = Chem.MolFromSmiles(smi).GetNumAtoms()

        assert isinstance(input_entry, datasets.MPNNEntry)
        assert isinstance(input_entry.target, torch.Tensor)
        assert isinstance(input_entry.node_features, torch.Tensor)
        assert input_entry.node_features.shape == (
            num_nodes + 1,
            constants.NUM_NODE_FEATURES,
        )
        assert isinstance(input_entry.edge_features, torch.Tensor)
        assert input_entry.edge_features.shape == (
            num_bonds,
            constants.NUM_EDGE_FEATURES,
        )
        assert input_entry.adj_matrix.shape == (num_nodes + 1, num_nodes + 1)
        assert input_entry.edge_indices.shape == (
            2,
            2 * num_bonds + 2 * num_nodes,
        )


if __name__ == "__main__":
    pytest.main([__file__])
