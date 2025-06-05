import numpy as np
from pyscf import dft, gto

from src.commons.types import Data, Mesh, OneElectron, TwoElectron
from src.dft.hamiltonian import Hamiltonian
from src.dft.orbital_features import tokenize_orbitals


def build_one_electron(mol: gto.Mole):
    # Compute one-electron integrals
    overlap = mol.intor("int1e_ovlp_cart").astype(np.float64)
    kinetic = mol.intor("int1e_kin_cart").astype(np.float64)
    nuclear = mol.intor("int1e_nuc_cart").astype(np.float64)

    # Organize data into named tuples
    return OneElectron(overlap=overlap, kinetic=kinetic, nuclear=nuclear)


def build_two_electron(mol: gto.Mole):
    # Compute two-electron integrals
    eri = mol.intor("int2e_cart", aosym="s1")
    return TwoElectron(eri=eri)


def build_mesh(mol: gto.Mole, grid_level: int = 3):
    # Generate grid for XC functional
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()

    points = grids.coords.astype(np.float64)
    weights = grids.weights.astype(np.float64)

    return Mesh(points=points, weights=weights)


def build_pyscf_mol(atomic_number, position, basis_name="sto-3g"):
    if not isinstance(atomic_number, np.ndarray):
        atomic_number = np.asarray(atomic_number)

    if not isinstance(position, np.ndarray):
        position = np.asarray(position)

    num_electrons = np.sum(atomic_number)

    mol = gto.Mole(unit="Bohr", spin=num_electrons % 2, cart=True)
    mol.atom = [(symbol, pos) for symbol, pos in zip(atomic_number, position)]
    mol.basis = basis_name
    mol.build(unit="Bohr")

    return mol


def build_molecule(
    atomic_number,
    position,
    energy=None,
    basis_name="sto-3g",
    xc_method="lda",
    grid_level: int = 3,
    center: bool = True,
):
    assert isinstance(atomic_number, np.ndarray) and isinstance(
        position, np.ndarray
    ), "PySCF expects numpy arrays as input"

    if center:
        position = position - position.mean(axis=0)[None, ...]

    mol = build_pyscf_mol(atomic_number, position, basis_name=basis_name)

    # Compute one-electron integrals
    one_electron = build_one_electron(mol)
    two_electron = build_two_electron(mol)
    mesh = build_mesh(mol, grid_level=grid_level)
    occ = np.full(mol.nao, 2.0)
    mask = occ.cumsum() > mol.nelectron
    occ = np.where(mask, 0.0, occ)

    hamiltonian = Hamiltonian(
        one_electron.kinetic,
        one_electron.nuclear,
        one_electron.overlap,
        eri=two_electron.eri,
        mesh=mesh,
        occupancy=occ,
        xc_method=xc_method,
        gridAO=mol.eval_gto("GTOval_cart_deriv1", mesh.points, 4),
    )

    # Map orbital to atom center and tokenize orbitals
    orbital_tokens, orbital_index = tokenize_orbitals(mol)

    return Data(
        atomic_number=atomic_number,
        position=position,
        energy=energy,
        hamiltonian=hamiltonian,
        orbital_tokens=orbital_tokens,
        orbital_index=orbital_index,
    )
