import math
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph


def cosine_cutoff(cutoff_lower: float, cutoff_upper: float) -> Callable:
    if cutoff_lower > 0:
        cutoff_fn = lambda distances: 0.5 * (
            jnp.cos(
                math.pi
                * (2 * (distances - cutoff_lower) / (cutoff_upper - cutoff_lower) + 1.0)
            )
            + 1.0
        )

    else:
        cutoff_fn = lambda distances: 0.5 * (
            jnp.cos(distances * math.pi / cutoff_upper) + 1.0
        )

    def _cosine_cutoff(distances):
        if cutoff_lower > 0:
            x = cutoff_fn(distances) * (distances < cutoff_upper).astype(float)
            x = x * (distances > cutoff_lower).astype(float)
            return x

        return cutoff_fn(distances) * (distances < cutoff_upper).astype(float)

    return _cosine_cutoff


def distance_fn(
    cutoff_lower: int = 0.0, cutoff_upper: int = 10.0, return_vecs: bool = False
):
    def _distance(position, senders, receivers):
        edge_vec = position[receivers] - position[senders]

        mask = receivers != senders
        edge_weight = jnp.zeros((edge_vec.shape[0],))
        edge_weight = jnp.linalg.norm(edge_vec, axis=-1) * mask

        lower_mask = edge_weight >= cutoff_lower

        senders = senders[lower_mask]
        receivers = receivers[lower_mask]
        edge_weight = edge_weight[lower_mask]

        if return_vecs:
            edge_vec = edge_vec[lower_mask]
            return senders, receivers, edge_weight, edge_vec

        return senders, receivers, edge_weight, None

    return _distance


class NeighborEmbedding(nn.Module):
    hidden_channels: int
    cutoff_lower: float
    cutoff_upper: float
    max_species: int = 100

    def setup(self):
        self.cutoff = cosine_cutoff(
            cutoff_lower=self.cutoff_lower, cutoff_upper=self.cutoff_upper
        )

    @nn.compact
    def __call__(self, z, x, senders, receivers, edge_weight, edge_attr):
        C = self.cutoff(edge_weight)
        W = nn.Dense(
            features=self.hidden_channels,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.constant(0),
        )(edge_attr)

        W = W * C.reshape(-1, 1)

        x_neighbors = nn.Embed(
            num_embeddings=self.max_species, features=self.hidden_channels
        )(z)

        graph = jraph.GraphsTuple(
            nodes=x_neighbors,
            senders=senders,
            receivers=receivers,
            edges=W,
            n_node=len(x_neighbors),
            n_edge=len(senders),
            globals=None,
        )

        x_neighbors = jraph.GraphNetwork(
            update_node_fn=self._update_fn, update_edge_fn=self._message
        )(graph).nodes

        return nn.Dense(self.hidden_channels)(jnp.concat([x, x_neighbors], axis=1))

    def _update_fn(
        self,
        nodes,
        senders,
        msg,
        globals_,
    ):
        return msg

    def _message(self, edge_features, incoming, outgoing, globals_) -> jnp.ndarray:
        _ = edge_features
        _ = globals_

        return edge_features * incoming


class GaussianSmearing(nn.Module):
    cutoff_lower: float = 0.0
    cutoff_upper = float = 10.0
    num_rbf: int = 50
    trainable: bool = True

    def setup(self):
        _init_offset = lambda rng, shape: jnp.linspace(
            self.cutoff_lower, self.cutoff_upper, self.num_rbf
        )
        _init_coeff = lambda rng, shape: -0.5 / (self.offset[1] - self.offset[0]) ** 2

        if self.trainable:
            self.offset = self.param("offset", _init_offset, (self.num_rbf,))

            self.coeff = self.param(
                "coeff",
                _init_coeff,
                (1,),
            )

        else:
            # input does not matter since initializers do not depend
            # on randomness nor shape
            self.offset = _init_offset(0, 0)
            self.coeff = _init_coeff(0, 0)

    def __call__(self, distance: jnp.ndarray):
        distance = distance[..., jnp.newaxis] - self.offset
        return jnp.exp(self.coeff * jnp.pow(distance, 2))


class ExpNormalSmearing(nn.Module):
    cutoff_lower: int = 0.0
    cutoff_upper: int = 5.0
    num_rbf: int = 50
    trainable: bool = True

    def setup(self):
        self.cutoff_fn = cosine_cutoff(self.cutoff_lower, self.cutoff_upper)
        self.alpha = 5.0 / (self.cutoff_upper - self.cutoff_lower)

        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = jnp.exp(-self.cutoff_upper + self.cutoff_lower)

        _init_means = lambda key, shape: jnp.linspace(start_value, 1, self.num_rbf)
        _init_betas = lambda key, shape: jnp.array(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )

        if self.trainable:
            self.means = self.param("means", _init_means, (self.num_rbf,))

            self.betas = self.param(
                "coeff",
                _init_betas,
                (self.num_rbf,),
            )

        else:
            # input does not matter since initializers do not depend
            # on randomness nor shape
            self.means = _init_means(0, 0)
            self.betas = _init_betas(0, 0)

    def __call__(self, distance):
        distance = distance[..., jnp.newaxis]
        return self.cutoff_fn(distance) * jnp.exp(
            -self.betas
            * (jnp.exp(self.alpha * (-distance + self.cutoff_lower)) - self.means) ** 2
        )


act_fn_map = {"silu": jax.nn.silu}
distance_expansion_map = {"gaussian": GaussianSmearing, "expnorm": ExpNormalSmearing}
