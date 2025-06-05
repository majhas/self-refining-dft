from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from src.commons.types import Graph


@partial(jax.jit, static_argnames=("apply_fn", "output_coefficients"))
def predict_energy(
    params: dict, apply_fn: callable, graph: Graph, output_coefficients: bool = False
) -> jnp.ndarray:
    """
    Vectorized forward pass to predict energies for a batched Graph.
    """
    return jax.vmap(
        lambda z, x, ot, oi, s, r, h: apply_fn(
            params,
            z,
            x,
            orbital_tokens=ot,
            orbital_index=oi,
            senders=s,
            receivers=r,
            hamiltonian=h,
            output_coefficients=output_coefficients,
        )
    )(
        graph.atomic_number,
        graph.position,
        graph.orbital_tokens,
        graph.orbital_index,
        graph.senders,
        graph.receivers,
        graph.hamiltonian,
    )


@partial(jax.jit, static_argnames=("apply_fn",))
def grad_energy(
    params: dict,
    apply_fn: callable,
    graph: Graph,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute energy and gradient wrt positions for each sample in a batch.
    """

    def _grad_energy_single(z, x, ot, oi, s, r, h):
        energy, grad = jax.value_and_grad(
            lambda _params, _z, _x: apply_fn(
                _params,
                _z,
                _x,
                orbital_tokens=ot,
                orbital_index=oi,
                senders=s,
                receivers=r,
                hamiltonian=h,
            ),
            argnums=2,
        )(params, z, x)
        return energy, grad

    energies, grads = jax.vmap(_grad_energy_single)(
        graph.atomic_number,
        graph.position,
        graph.orbital_tokens,
        graph.orbital_index,
        graph.senders,
        graph.receivers,
        graph.hamiltonian,
    )
    return energies, grads
