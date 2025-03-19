from rdkit import Chem
import torch


def featurize_atoms(molecule: Chem.Mol) -> torch.Tensor:
    atoms = molecule.GetAtoms()
    features = []
    for atom in atoms:
        atomic_number = atom.GetAtomicNum()
        degree = atom.GetDegree()
        valence = atom.GetImplicitValence()
        num_hydrogens = atom.GetTotalNumHs()
        is_aromatic = atom.GetIsAromatic()
        features.append([atomic_number, degree, valence,
                        num_hydrogens, is_aromatic])
    return torch.as_tensor(features)
