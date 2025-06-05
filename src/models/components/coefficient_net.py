import flax.linen as nn
import jax
import jax.numpy as jnp

from src.models.components.attention import QuantumBiasedAttention


class CoefficientNetwork(nn.Module):
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
    def __call__(self, x, orbital_tokens, orbital_index, hamiltonian):
        x = x[orbital_index]

        orbital_emb = nn.Embed(
            num_embeddings=self.max_species, features=self.hidden_channels
        )(orbital_tokens)

        Z = jnp.eye(orbital_tokens.shape[0])
        C = hamiltonian.orthonormalise(Z)
        P = hamiltonian.density_matrix(C)
        for i in range(self.num_layers):
            P_init = P
            orbital_emb = nn.Sequential(
                [
                    nn.Dense(self.hidden_channels),
                    jax.nn.silu,
                    nn.Dense(self.hidden_channels),
                ]
            )(orbital_emb)

            x = x + orbital_emb

            old_x = x

            x = QuantumBiasedAttention(hidden_channels=self.hidden_channels)(
                x, hamiltonian, P_init
            )

            if self.resnet:
                x = x + old_x

            x = jax.nn.silu(x)

            qk = nn.Dense(self.hidden_channels * 2, use_bias=False)(x)
            q, k = jnp.split(qk, 2, axis=-1)

            Z = jnp.einsum("ih,jh->ij", q, k) / jnp.sqrt(self.hidden_channels)

            if self.norm_coefficients:
                Z = Z / jnp.linalg.norm(Z, keepdims=True)

            C = hamiltonian.orthonormalise(Z)
            P = hamiltonian.density_matrix(C)

            if self.density_mixing and i != (self.num_layers - 1):
                P = 0.5 * P + 0.5 * P_init

        return C
