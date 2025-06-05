"""Adapted from MESS https://github.com/valence-labs/mess """

import equinox as eqx
import jax.numpy as jnp
import jax.numpy.linalg as jnl
from jaxtyping import Array, Float

from src.commons.types import Mesh
from src.dft.orthnorm import symmetric
from src.dft.xc import b3lyp

EPSILON_B3LYP = 1e-20
HYB_B3LYP = 0.2
FloatNxN = Float[Array, "N N"]


def exchange_correlation(density_matrix, grid_AO, grid_weights):
    grid_AO_dm = grid_AO[0] @ density_matrix  # (gsize, N) @ (N, N) -> (gsize, N)
    grid_AO_dm = jnp.expand_dims(grid_AO_dm, axis=0)  # (1, gsize, N)
    mult = grid_AO_dm * grid_AO
    rho = jnp.sum(mult, axis=2)  # (4, grid_size)=(4, 45624) for C6H6.
    E_xc, vrho, vgamma = b3lyp(
        rho, EPSILON_B3LYP
    )  # (gridsize,) (gridsize,) (gridsize,)
    E_xc = jnp.sum(
        rho[0] * grid_weights * E_xc
    )  # float=-27.968[Ha] for C6H6 at convergence.
    rho = (
        jnp.concatenate([vrho.reshape(1, -1) / 2, 4 * vgamma * rho[1:4]], axis=0)
        * grid_weights
    )  # (4, grid_size)=(4, 45624)
    grid_AO_T = grid_AO[0].T  # (N, gsize)
    rho = jnp.expand_dims(rho, axis=2)  # (4, gsize, 1)
    grid_AO_rho = grid_AO * rho  # (4, gsize, N)
    sum_grid_AO_rho = jnp.sum(grid_AO_rho, axis=0)  # (gsize, N)
    V_xc = grid_AO_T @ sum_grid_AO_rho  # (N, N)
    V_xc = V_xc + V_xc.T  # (N, N)
    return E_xc, V_xc


def get_JK(density_matrix, ERI):
    """Computes the (N, N) matrices J and K. Density matrix is (N, N) and ERI is (N, N, N, N)."""
    N = density_matrix.shape[0]

    J = jnp.einsum("ijkl,ji->kl", ERI, density_matrix)  # (N, N)
    K = jnp.einsum("ijkl,jk->il", ERI, density_matrix)  # (N, N)
    diff_JK = J - (K / 2 * HYB_B3LYP)

    diff_JK = diff_JK.reshape(N, N)

    return diff_JK


class Hamiltonian(eqx.Module):
    H_core: jnp.ndarray
    X: jnp.ndarray
    eri: jnp.ndarray
    mesh: Mesh
    occupancy: jnp.ndarray
    gridAO: jnp.ndarray
    xcfunc: eqx.Module

    def __init__(
        self,
        kinetic,
        nuclear,
        overlap,
        eri,
        mesh,
        occupancy,
        gridAO,
        xc_method,
        ont=symmetric,
    ):
        self.X = ont(overlap)
        self.H_core = kinetic + nuclear
        self.eri = eri
        self.mesh = mesh
        self.gridAO = gridAO
        self.occupancy = occupancy
        self.xcfunc = 0

    def __call__(self, P):
        E_core = jnp.sum(self.H_core * P)
        diff_JK = get_JK(P, self.eri)
        E_JK = jnp.sum(P * diff_JK)
        E_xc, _ = exchange_correlation(P, self.gridAO, self.mesh.weights)

        E = E_core + 0.5 * E_JK + E_xc
        return E

    def density_matrix(self, C):
        return jnp.einsum("k,ik,jk->ij", self.occupancy, C, C)

    def orthonormalise(self, Z):
        C = self.X @ jnl.qr(Z).Q
        return C

    def coloumb(self, P: FloatNxN) -> FloatNxN:
        """Build the Coloumb matrix (classical electrostatic) from the density matrix.

        Args:
            P (FloatNxN): the density matrix

        Returns:
            FloatNxN: Coloumb matrix
        """
        return jnp.einsum("kl,ijkl->ij", P, self.eri)

    def exchange(self, P: FloatNxN) -> FloatNxN:
        """Build the quantum-mechanical exchange matrix from the density matrix

        Args:
            P (FloatNxN): the density matrix

        Returns:
            FloatNxN: Exchange matrix
        """
        return jnp.einsum("ij,ikjl->kl", P, self.eri)

    @classmethod
    def from_precomputed(cls, H_core, X, eri, mesh, occupancy, gridAO):
        # Directly set attributes without calling __init__
        dummy_instance = cls(
            kinetic=jnp.zeros((0,)),
            nuclear=jnp.zeros((0,)),
            overlap=jnp.zeros((0,)),
            eri=jnp.zeros((0, 0, 0, 0)),
            mesh=Mesh(points=jnp.zeros((0, 3)), weights=jnp.zeros(0)),  # Empty mesh
            occupancy=jnp.zeros((0,)),
            gridAO=jnp.zeros((0,)),
            ont=lambda x: x,  # Identity function as a placeholder
            xc_method="b3lyp",  # Default xc_method
        )

        # Use eqx.tree_at to update the attributes immutably
        updated_instance = eqx.tree_at(
            lambda obj: (
                obj.H_core,
                obj.X,
                obj.eri,
                obj.mesh,
                obj.occupancy,
                obj.gridAO,
                obj.xcfunc,
            ),
            dummy_instance,
            (H_core, X, eri, mesh, occupancy, gridAO, 0),
        )

        return updated_instance
