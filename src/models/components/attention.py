import flax.linen as nn
import jax.numpy as jnp


def pairwise_distance(x, eps: float = 1e-4):
    diff = (x[None, ...] - x[:, None, ...]) ** 2
    dist = jnp.sqrt(diff.sum(axis=-1) + eps)
    return dist


class QuantumBiasedAttention(nn.Module):
    hidden_channels: int = 128
    num_heads: int = 16
    activation: str = "silu"

    def setup(self):
        self.head_dim: int = self.hidden_channels // self.num_heads

    @nn.compact
    def __call__(self, x, hamiltonian, P_init):
        H_core = hamiltonian.H_core
        X = hamiltonian.X
        J = jnp.einsum("kl,ijkl->ij", P_init, hamiltonian.eri)
        K = jnp.einsum("ij,ikjl->kl", P_init, hamiltonian.eri)

        diff_JK = J - K / 2
        L = X @ H_core @ X.T

        num_orbitals = x.shape[0]

        x = nn.LayerNorm()(x)
        shape = (num_orbitals, self.num_heads, self.head_dim)
        qkv = nn.Dense(self.hidden_channels * 3, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = [arr.reshape(shape) for arr in (q, k, v)]

        qk = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(self.head_dim)
        qk = qk.at[0:2, ...].add(H_core[None, ...])
        qk = qk.at[2:4, ...].add(diff_JK[None, ...])
        qk = qk.at[4:6, ...].add(L[None, ...])

        qk = nn.softmax(qk, axis=-1)

        qkv = jnp.einsum("hij,jhd->ihd", qk, v).reshape(num_orbitals, -1)
        out = nn.Dense(self.hidden_channels)(qkv)

        return out


class MultiHeadAttention(nn.Module):
    hidden_channels: int = 128
    num_heads: int = 16
    activation: str = "silu"
    default_bias: bool = True

    def setup(self):
        self.head_dim: int = self.hidden_channels // self.num_heads

    @nn.compact
    def __call__(self, x):
        num_orbitals = x.shape[0]

        x = nn.LayerNorm()(x)
        shape = (num_orbitals, self.num_heads, self.head_dim)
        qkv = nn.Dense(self.hidden_channels * 3, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = [arr.reshape(shape) for arr in (q, k, v)]

        qk = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(self.head_dim)
        qk = nn.softmax(qk, axis=-1)

        qkv = jnp.einsum("hij,jhd->ihd", qk, v).reshape(num_orbitals, -1)
        out = nn.Dense(self.hidden_channels)(qkv)

        return out
