import flax.linen as nn
import jax.numpy as jnp

from src.networks.etnn.modules import (
    EquivariantMultiHeadAttention,
    GatedEquivariantBlock,
)
from src.networks.etnn.utils import (
    NeighborEmbedding,
    distance_expansion_map,
    distance_fn,
)


class EquivariantTransformerBackBone(nn.Module):
    hidden_channels: int = 128
    out_channels: int = None
    num_layers: int = 8
    num_rbf: int = 64
    rbf_type: str = "expnorm"
    trainable_rbf: bool = False
    activation: str = "silu"
    neighbor_embedding: bool = True
    cutoff_lower: float = 0.0
    cutoff_upper: float = 10.0
    max_species: int = 100
    node_attr_dim: int = 0
    edge_attr_dim: int = 0
    attn_activation: str = "silu"
    num_heads: int = 8
    clip_during_norm: bool = True
    qk_norm: bool = True
    norm_coors: bool = True
    norm_coors_scale_init: float = 1e-2

    def setup(self):
        self.distance_fn = distance_fn(
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            return_vecs=True,
        )

    @nn.compact
    def __call__(self, atomic_number, position, senders, receivers, eps=1e-6):
        x = nn.Embed(self.max_species, self.hidden_channels)(atomic_number)
        edge_vec = position[receivers] - position[senders]
        edge_weight = jnp.linalg.norm(edge_vec + eps, axis=-1)

        edge_attr = distance_expansion_map[self.rbf_type](
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            num_rbf=self.num_rbf,
            trainable=self.trainable_rbf,
        )(edge_weight)

        mask = senders == receivers
        masked_edge_weight = ((edge_weight * ~mask) + mask)[:, jnp.newaxis]

        if self.clip_during_norm:
            masked_edge_weight = masked_edge_weight.clip(min=1e-2)

        edge_vec = edge_vec / masked_edge_weight
        edge_weight = edge_weight[..., jnp.newaxis]

        x = NeighborEmbedding(
            hidden_channels=self.hidden_channels,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            max_species=self.max_species,
        )(
            z=atomic_number,
            x=x,
            senders=senders,
            receivers=receivers,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
        )

        vec = jnp.zeros((x.shape[0], 3, self.hidden_channels))

        for _ in range(self.num_layers):
            dx, dvec = EquivariantMultiHeadAttention(
                hidden_channels=self.hidden_channels,
                num_rbf=self.num_rbf,
                num_heads=self.num_heads,
                activation=self.activation,
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                qk_norm=self.qk_norm,
                norm_coors=self.norm_coors,
                norm_coors_scale_init=self.norm_coors_scale_init,
            )(
                x=x,
                vec=vec,
                senders=senders,
                receivers=receivers,
                edge_weight=edge_weight,
                edge_attr=edge_attr,
                edge_vec=edge_vec,
            )

            x = x + dx
            vec = vec + dvec

        x = nn.LayerNorm(self.hidden_channels)(x)

        return x, vec


class EquivariantTransformer(nn.Module):
    hidden_channels: int = 128
    num_layers: int = 8
    num_rbf: int = 64
    rbf_type: str = "expnorm"
    trainable_rbf: bool = False
    activation: str = "silu"
    neighbor_embedding: bool = True
    cutoff_lower: float = 0.0
    cutoff_upper: float = 10.0
    max_species: int = 100
    node_attr_dim: int = 0
    edge_attr_dim: int = 0
    attn_activation: str = "silu"
    num_heads: int = 8
    clip_during_norm: bool = True
    qk_norm: bool = True
    norm_coors: bool = True
    norm_coors_scale_init: float = 1e-2

    @nn.jit
    @nn.compact
    def __call__(self, atomic_number, position, senders, receivers):
        x, vec = EquivariantTransformerBackBone(
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            num_rbf=self.num_rbf,
            rbf_type=self.rbf_type,
            trainable_rbf=self.trainable_rbf,
            activation=self.activation,
            neighbor_embedding=self.neighbor_embedding,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            max_species=self.max_species,
            node_attr_dim=self.node_attr_dim,
            edge_attr_dim=self.edge_attr_dim,
            attn_activation=self.attn_activation,
            num_heads=self.num_heads,
            clip_during_norm=self.clip_during_norm,
            qk_norm=self.qk_norm,
            norm_coors=self.norm_coors,
            norm_coors_scale_init=self.norm_coors_scale_init,
        )(
            atomic_number=atomic_number,
            position=position,
            senders=senders,
            receivers=receivers,
        )

        output_blocks = [
            GatedEquivariantBlock(
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels // 2,
                activation=self.activation,
                scalar_activation=True,
                vector_output=True,
            ),
            GatedEquivariantBlock(
                hidden_channels=self.hidden_channels // 2,
                out_channels=self.hidden_channels,
                activation=self.activation,
            ),
        ]

        for block in output_blocks:
            x, vec = block(x=x, vec=vec)

        return x, vec
