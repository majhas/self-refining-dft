import numpy as np
from pyscf import gto

ORBITAL_TOKENS = {
    "1s": 0,
    "2s": 1,
    "2px": 2,
    "2py": 3,
    "2pz": 4,
    "3s": 5,
    "3px": 6,
    "3py": 7,
    "3pz": 8,
    "3dxx": 9,
    "3dxy": 10,
    "3dxz": 11,
    "3dyy": 12,
    "3dyz": 13,
    "3dzz": 14,
    "4s": 15,
    "4px": 16,
    "4py": 17,
    "4pz": 18,
    "4dxx": 19,
    "4dxy": 20,
    "4dxz": 21,
    "4dyy": 22,
    "4dyz": 23,
    "4dzz": 24,
    "4fxxx": 25,
    "4fxxy": 26,
    "4fxxz": 27,
    "4fxyy": 28,
    "4fxyz": 29,
    "4fyyz": 30,
    "4fzzz": 31,
    "5s": 32,
    "5px": 33,
    "5py": 34,
    "5pz": 35,
    "5dxx": 36,
    "5dxy": 37,
    "5dxz": 38,
    "5dyy": 39,
    "5dyz": 40,
    "5dzz": 41,
    "5fxxx": 42,
    "5fxxy": 43,
    "5fxxz": 44,
    "5fxyy": 45,
    "5fxyz": 46,
    "5fyyz": 47,
    "5fzzz": 48,
}


def tokenize_orbitals(mol: gto.Mole):
    ao_labels = mol.ao_labels()
    orbital_tokens = []
    orbital_index = []

    for label in ao_labels:
        label = label.strip().split(" ")
        index = label[0]
        orbital = label[-1]
        orbital_index.append(int(index))
        orbital_tokens.append(ORBITAL_TOKENS[orbital])

    orbital_tokens = np.array(orbital_tokens)
    orbital_index = np.array(orbital_index)
    return orbital_tokens, orbital_index
