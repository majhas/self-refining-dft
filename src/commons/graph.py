from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from src.commons.types import Graph, MolGraph
from src.dft.molecule import build_molecule


def fully_connected_edges(num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate senders and receivers for a fully-connected graph without self-loops.
    """
    # Create a binary adjacency matrix without self-loops
    adjacency = 1 - np.eye(num_nodes, dtype=int)
    senders, receivers = np.where(adjacency)
    return senders, receivers


def compute_edge_distances(
    position: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distances between connected node pairs.

    Returns:
        Array of shape (num_edges, 1) with distances.
    """
    diffs = position[receivers] - position[senders]
    distances = np.linalg.norm(diffs, axis=-1, keepdims=True)
    return distances


def build_graph(
    atomic_number: np.ndarray,
    position: np.ndarray,
    energy: float = 0.0,
    # if these three are all None, we’ll call build_molecule
    hamiltonian=None,
    orbital_index: np.ndarray = None,
    orbital_tokens: np.ndarray = None,
    # otherwise use these args with pyscf
    basis_name: str = "sto-3g",
    xc_method: str = "lda",
    grid_level: int = 3,
    center: bool = True,
) -> Graph:
    if center:
        position = position - np.mean(position, axis=0, keepdims=True)

    num_nodes = position.shape[0]
    senders, receivers = fully_connected_edges(num_nodes)
    edge_features = compute_edge_distances(position, senders, receivers)

    if any(x is None for x in (hamiltonian, orbital_index, orbital_tokens)):
        data = build_molecule(
            atomic_number=atomic_number,
            position=position,
            energy=energy,
            basis_name=basis_name,
            xc_method=xc_method,
            grid_level=grid_level,
            center=center,
        )
        hamiltonian = data.hamiltonian
        orbital_index = data.orbital_index
        orbital_tokens = data.orbital_tokens

    energy = np.array(energy).reshape(1)
    return Graph(
        atomic_number=atomic_number,
        position=position,
        energy=energy,
        edge_features=edge_features,
        senders=senders,
        receivers=receivers,
        hamiltonian=hamiltonian,
        orbital_index=orbital_index,
        orbital_tokens=orbital_tokens,
    )


def build_graph_no_hamil(atomic_number, position, energy=0.0, center=True):
    """
    Build a basic MolGraph without Hamiltonian or orbital features.

    Args:
        atomic_number: Array of atomic numbers.
        position: Atom coordinates.
        energy: System energy.
        center: Whether to center position.

    Returns:
        A MolGraph instance with edge distances.
    """
    if center:
        position = position - np.mean(position, axis=0, keepdims=True)

    senders, receivers = fully_connected_edges(position)
    edge_features = compute_edge_distances(
        position=position, senders=senders, receivers=receivers
    )

    return MolGraph(
        atomic_number=atomic_number,
        position=position,
        energy=np.array(energy),
        edge_features=edge_features,
        senders=senders,
        receivers=receivers,
    )


def batch_data(data):
    if not isinstance(data, list):
        data = [data]

    return jax.tree.map(lambda *xs: jnp.stack(jnp.array(xs)), *data)
