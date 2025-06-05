from typing import Any, Union

import flax
import flax.struct
import jax.numpy as jnp
import numpy as np


@flax.struct.dataclass
class OneElectron:
    overlap: Any
    kinetic: Any
    nuclear: Any


@flax.struct.dataclass
class TwoElectron:
    eri: Any


@flax.struct.dataclass
class Mesh:
    points: Any
    weights: Any


@flax.struct.dataclass
class Data:
    atomic_number: Union[np.array, jnp.array]
    position: Union[np.array, jnp.array]
    orbital_tokens: Union[np.array, jnp.array]
    orbital_index: Union[np.array, jnp.array]
    hamiltonian: Any = None
    energy: Union[np.array, jnp.array, float] = None


@flax.struct.dataclass
class Graph:
    atomic_number: Union[np.array, jnp.array]
    position: Union[np.array, jnp.array]
    edge_features: Union[np.array, jnp.array]
    senders: Union[np.array, jnp.array]
    receivers: Union[np.array, jnp.array]
    orbital_tokens: Union[np.array, jnp.array]
    orbital_index: Union[np.array, jnp.array]
    hamiltonian: Any = 0
    energy: Union[np.array, jnp.array, float] = 0


@flax.struct.dataclass
class MolGraph:
    atomic_number: Union[np.array, jnp.array]
    position: Union[np.array, jnp.array]
    edge_features: Union[np.array, jnp.array]
    senders: Union[np.array, jnp.array]
    receivers: Union[np.array, jnp.array]
    energy: Union[np.array, jnp.array, float] = 0
