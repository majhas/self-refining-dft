import jax.numpy as jnp
import jax.numpy.linalg as jnl
import numpy as np
from pyscf.dft.rks import RKS

from src.dft.hamiltonian import exchange_correlation, get_JK
from src.dft.molecule import build_pyscf_mol


def fock_matrix(H, C):
    P = H.density_matrix(C)
    diff_JK = get_JK(P, H.eri)
    _, V_xc = exchange_correlation(P, H.gridAO, H.mesh.weights)
    return H.H_core + diff_JK + V_xc


def electronic_energy(H, C):
    P = H.density_matrix(C)
    return H(P)


def nuclear_energy(atomic_number, position):
    charge = atomic_number[jnp.newaxis, ...] * atomic_number[..., jnp.newaxis]
    dist = jnp.linalg.norm(
        position[jnp.newaxis, ...] - position[:, jnp.newaxis, ...] + 1e-6, axis=-1
    )
    # dist = jnp.where(dist > 0, dist, 1.)
    return jnp.triu(charge / dist, k=1).sum()


def total_energy(H, C, atomic_number, position):
    E_elec = electronic_energy(H, C)
    E_nuc = nuclear_energy(atomic_number, position)
    return E_elec + E_nuc


def orbital_energy(H, C):
    F = fock_matrix(H, C)
    eigval = jnl.eigh(H.X.T @ F @ H.X)[0]
    return eigval


def homo_lumo_gap(H, C):
    eps = orbital_energy(H, C)
    occ_mask = H.occupancy > 0
    # HOMO: highest eps where occupied
    eps_h = eps[jnp.where(occ_mask, eps, -jnp.inf).argmax()]
    # LUMO: lowest eps where unoccupied
    eps_l = eps[jnp.where(~occ_mask, eps, jnp.inf).argmin()]
    return eps_h, eps_l, eps_l - eps_h


def run_pyscf_solver(
    atomic_number, position, basis_name="sto-3g", xc_method="b3lyp", max_cycle=15000
):
    mol = build_pyscf_mol(
        np.array(atomic_number), np.array(position), basis_name=basis_name
    )
    mf = RKS(mol, xc=xc_method)
    mf.max_cycle = max_cycle
    mf.run()

    return mf.energy_tot(), mf.mo_coeff
