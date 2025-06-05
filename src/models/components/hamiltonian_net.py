import jax
import jax.numpy as jnp
from flax import linen as nn

from src.models.components.attention import MultiHeadAttention


class HamiltonianNetwork(nn.Module):
    hidden_channels: int = 128
    num_layers: int = 4
    max_species: int = 20
    num_heads: int = 8
    activation: str = "silu"
    norm_coefficients: bool = False
    resnet: bool = False
    density_mixing: bool = False
    default_bias: bool = True

    @nn.compact
    def __call__(self, x, orbital_tokens, orbital_index):
        x = x[orbital_index]

        orbital_emb = nn.Embed(
            num_embeddings=self.max_species, features=self.hidden_channels
        )(orbital_tokens)

        for i in range(self.num_layers):
            orbital_emb = nn.Sequential(
                [
                    nn.Dense(self.hidden_channels),
                    jax.nn.silu,
                    nn.Dense(self.hidden_channels),
                ]
            )(orbital_emb)

            x = x + orbital_emb

            old_x = x

            x = MultiHeadAttention(
                hidden_channels=self.hidden_channels, default_bias=self.default_bias
            )(x)

            if self.resnet:
                x = x + old_x

            x = jax.nn.silu(x)

            qk = nn.Dense(self.hidden_channels * 2, use_bias=False)(x)
            q, k = jnp.split(qk, 2, axis=-1)

            H = jnp.einsum("ih,jh->ij", q, k) / jnp.sqrt(self.hidden_channels)

        return H
