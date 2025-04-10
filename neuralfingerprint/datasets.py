import torch
from rdkit import Chem
from torch.utils import data as torch_data
from neuralfingerprint import featurizer


class NeuralFingerprintDataset(torch_data.Dataset):
    """Creates a molecule dataset based on neural fingerprints."""

    def __init__(self, smiles: tuple[str, ...], targets: tuple[float, ...]):
        self.smiles = smiles
        self.targets = targets

    def __len__(self):
        return len(self.smiles)

    def transform(self, smiles: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Featurize one molecule.

        Args:
            smiles: The input SMILES to be featurized
        Returns:
            A tuple with atom and bond features, and the prediction target.
        """
        mol = Chem.MolFromSmiles(smiles)

        atom_features = featurizer.featurize_atoms(mol).to(torch.float32)
        bond_features = featurizer.featurize_bonds(mol).to(torch.float32)

        return atom_features, bond_features

    def __getitem__(self, idx) -> tuple[torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor]:
        """Returns one element from the dataset.

        Args:
            idx: index to retrieve.

        Returns:
            A tuple with atom and bond features, and the prediction target.


        """
        atom_features, bond_features = self.transform(self.smiles[idx])
        target = self.targets[idx]
        return atom_features, bond_features, target
