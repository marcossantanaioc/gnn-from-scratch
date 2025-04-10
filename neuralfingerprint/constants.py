"""Store neuralfingerprint constants"""

from typing import Final

from rdkit.Chem import rdchem

MAX_DEGREE: Final[int] = 6
MAX_VALENCE: Final[int] = 6
MAX_HYDROGEN_ATOMS: Final[int] = 5
MAX_ATOMIC_NUMBER: Final[int] = 118
MAX_BOND_TYPES: Final[int] = 22
ADDITIONAL_BOND_FEATURES: Final[int] = 2  # Conjugation and is in ring

# Define number of features
NUM_BOND_FEATURES: Final[int] = MAX_BOND_TYPES + ADDITIONAL_BOND_FEATURES
NUM_ATOM_FEATURES: Final[int] = (
    MAX_DEGREE + MAX_VALENCE + MAX_HYDROGEN_ATOMS + MAX_ATOMIC_NUMBER
)


BOND_TYPES: dict[rdchem.BondType, int] = {
    v: k for k, v in rdchem.BondType.values.items()
}
