from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from src.commons.graph import build_graph
from src.commons.types import Graph
from src.models.components.energy import grad_energy, predict_energy


def build_graph_batch(
    atomic_number: np.ndarray,
    positions: np.ndarray,
    basis_name: str,
    xc_method: str,
    grid_level: int,
) -> Graph:
    """
    Given numpy arrays of shape (batch, ...), build a batched Graph by calling
    build_graph per sample, then stacking fields via jax.tree_map.
    """
    graphs = [
        build_graph(
            atomic_number=atomic_number[i],
            position=positions[i],
            basis_name=basis_name,
            xc_method=xc_method,
            grid_level=grid_level,
        )
        for i in range(len(positions))
    ]
    # Stack all Graph fields into batch dim
    return jax.tree_map(lambda *xs: jnp.stack(xs), *graphs)


def update_step_size(accept_rate: float, step: float) -> float:
    if accept_rate > 0.55:
        return step * 1.1
    if accept_rate < 0.5:
        return step / 1.1
    return step


def rwmh_step(
    key: jax.random.PRNGKey,
    params: dict,
    apply_fn: callable,
    graph: Graph,
    step_size: float = 1.0,
    basis_name: str = "sto-3g",
    xc_method: str = "lda",
    grid_level: int = 3,
):
    _, *loc_keys = jax.random.split(key, 3)
    position = graph.position
    proposal = position + step_size * jax.random.normal(loc_keys[0], position.shape)

    proposal_graph = build_graph_batch(
        np.array(graph.atomic_number),
        np.array(proposal),
        basis_name,
        xc_method,
        grid_level,
    )
    logp_proposal = -predict_energy(params, apply_fn, proposal_graph).reshape(-1, 1)
    logp_current = -predict_energy(params, apply_fn, graph).reshape(-1, 1)
    u = jax.random.uniform(loc_keys[1], logp_proposal.shape)
    mask = (logp_proposal - logp_current) > jnp.log(u)
    mask = mask[..., None]
    x_next = mask * proposal + (1 - mask) * position
    accept_rate = mask.mean()

    return jax.lax.stop_gradient(x_next), accept_rate


def mala_step(
    key: jax.random.PRNGKey,
    params: dict,
    apply_fn: callable,
    graph: Graph,
    step_size: float,
    basis_name: str = "sto-3g",
    xc_method: str = "lda",
    grid_level: int = 3,
) -> Tuple[jnp.ndarray, float]:
    x = graph.position
    _, *loc_keys = jax.random.split(key, 3)
    energy_curr, dUdx = grad_energy(params=params, apply_fn=apply_fn, graph=graph)

    noise = jax.random.normal(loc_keys[0], x.shape)
    proposal = x - step_size * dUdx + jnp.sqrt(2 * step_size) * noise

    prop_graph = build_graph_batch(
        np.array(graph.atomic_number),
        np.array(proposal),
        basis_name,
        xc_method,
        grid_level,
    )

    energy_prop, dUdx_prop = grad_energy(
        params=params, apply_fn=apply_fn, graph=prop_graph
    )

    def logq(xa, xb, grad):
        # grad = - dUdx = \nabla logp(x)
        diff = xa - xb + step_size * grad
        return -0.5 * jnp.sum(diff**2, axis=(-1, -2)) / (2 * step_size)

    # logp(x) = - E(x)
    logp = -energy_prop + energy_curr
    logp += logq(x, proposal, dUdx_prop)
    logp -= logq(proposal, x, dUdx)
    logp = logp.reshape(-1, 1)

    u = jax.random.uniform(loc_keys[1], shape=logp.shape)
    mask = logp > jnp.log(u)
    mask = mask[:, None]
    x_next = mask * proposal + (1 - mask) * x
    accept_rate = mask.mean()

    return jax.lax.stop_gradient(x_next), accept_rate


@partial(jax.jit, static_argnames=("apply_fn"))
def ula_step(
    key: jax.random.PRNGKey,
    params: dict,
    apply_fn: callable,
    graph: Graph,
    step_size: float,
) -> jnp.ndarray:
    _, loc_key = jax.random.split(key)
    _, dUdx = grad_energy(params, apply_fn, graph)
    noise = jax.random.normal(loc_key, graph.position.shape)
    x_next = graph.position - step_size * dUdx + jnp.sqrt(2 * step_size) * noise
    return jax.lax.stop_gradient(x_next), 1.0


def run_mcmc(
    key: jax.random.PRNGKey,
    params: dict,
    apply_fn: callable,
    atomic_number: np.ndarray,
    init_pos: Optional[np.ndarray] = None,
    step_size: float = 1e-2,
    n_iter: int = 100,
    batch_size: int = 4,
    num_samples: int = 4,
    basis_name: str = "sto-3g",
    xc_method: str = "lda",
    grid_level: int = 3,
    kernel: str = "mala",
    return_chain: bool = False,
) -> Tuple[Graph, float, float, Optional[np.ndarray]]:
    num_atoms = atomic_number.shape[0]

    # prepare data
    if init_pos is None:
        init_pos = np.random.randn(num_samples, num_atoms, 3)
    else:
        assert init_pos.ndim == 3, (
            "initial positions for MCMC must be of form (batch_size, num_atoms, 3)"
            f" but got init_pos ndim == {init_pos.ndim}"
        )
        num_samples = init_pos.shape[0]

    position = init_pos.copy()
    if atomic_number.ndim < 2:
        atomic_number = atomic_number[None, ...]

    if atomic_number.shape[0] != num_samples:
        atomic_number = atomic_number.repeat(num_samples, axis=0)

    # build_graph_batch expects numpy array for PySCF
    atomic_number = np.array(atomic_number)
    position = np.array(position)
    graph = build_graph_batch(
        atomic_number, position, basis_name, xc_method, grid_level
    )
    accept_rates = []
    chain = [] if return_chain else None

    # select kernel
    if kernel == "rwmh":
        step_fn = partial(
            rwmh_step, basis_name=basis_name, xc_method=xc_method, grid_level=grid_level
        )
    elif kernel == "mala":
        step_fn = partial(
            mala_step, basis_name=basis_name, xc_method=xc_method, grid_level=grid_level
        )
    elif kernel == "ula":
        step_fn = partial(ula_step)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    for i in range(n_iter):
        key, loc_key = jax.random.split(key)
        position, accept_rate = step_fn(loc_key, params, apply_fn, graph, step_size)
        accept_rates.append(accept_rate)

        graph = build_graph_batch(
            np.array(atomic_number),
            np.array(position),
            basis_name,
            xc_method,
            grid_level,
        )
        if return_chain:
            chain.append(position[:, None, ...])

        if kernel == "mala" or kernel == "rwmh":
            step_size = update_step_size(accept_rate, step_size)

    avg_accept_rate = float(jnp.mean(jnp.stack(accept_rates)))
    energies = predict_energy(params, apply_fn, graph)
    graph = graph.replace(energy=energies)

    if return_chain:
        chain_arr = np.stack(chain, axis=1)
        return graph, step_size, avg_accept_rate, chain_arr

    return graph, step_size, avg_accept_rate
