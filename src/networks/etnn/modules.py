import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from src.networks.etnn.utils import act_fn_map, cosine_cutoff


class CoorsNorm(nn.Module):
    eps: float = 1e-8
    scale_init: float = 1.0

    @nn.compact
    def __call__(self, coors_feature: jnp.ndarray):
        """Coordinate features normalization"""

        scale = self.param("scale", lambda key, shape: self.scale_init, (1,))

        # shape of coors_feature: (num_atoms, 3, hidden_channels)
        norm = jnp.linalg.norm(coors_feature, axis=1, keepdims=True)
        normed_coords = coors_feature / norm.clip(min=self.eps)
        return normed_coords * scale


class GatedEquivariantBlock(nn.Module):
    hidden_channels: int
    out_channels: int = None
    activation: str = "silu"
    scalar_activation: bool = False
    vector_output: bool = False
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, vec):
        out_channels = (
            self.hidden_channels if self.out_channels is None else self.out_channels
        )
        vec2_out_channels = self.out_channels if self.vector_output else 1

        vec1_buffer = nn.Dense(
            self.hidden_channels,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
            name="vec1_buffer_dense",
        )(vec)

        vec1 = jnp.zeros((vec1_buffer.shape[0], vec1_buffer.shape[2]))
        mask = (vec1_buffer != 0).reshape(vec1_buffer.shape[0], -1).all(axis=1)
        mask = mask[..., jnp.newaxis].astype(int)
        vec1 = (jnp.linalg.norm(vec1_buffer + self.eps, axis=-2)) * mask

        vec2 = nn.Dense(
            vec2_out_channels,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
        )(vec)

        act_fn = act_fn_map[self.activation]

        dense_w_init = Partial(
            nn.Dense,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.constant(0.0),
        )

        x = jnp.concat([x, vec1], axis=-1)

        if self.vector_output:
            out_channels += out_channels
        else:
            out_channels += 1

        out = nn.Sequential(
            [dense_w_init(self.hidden_channels), act_fn, dense_w_init(out_channels)]
        )(x)

        x, vec = out[:, : self.out_channels], out[:, self.out_channels :]
        vec = vec[:, jnp.newaxis] * vec2

        if self.scalar_activation:
            x = act_fn(x)

        return x, vec


class EquivariantMultiHeadAttention(nn.Module):
    hidden_channels: int
    num_rbf: int
    num_heads: int
    activation: str
    attn_activation: str
    cutoff_lower: float
    cutoff_upper: float
    qk_norm: bool = True
    norm_coors: bool = False
    norm_coors_scale_init: float = 1e-2

    def setup(self):
        self.head_dim = self.hidden_channels // self.num_heads
        self.cutoff_fn = cosine_cutoff(self.cutoff_lower, self.cutoff_upper)

    def _message(
        self,
        q_i,
        k_j,
        v_j,
        vec_j,
        dk,
        dv,
        edge_weight,
        edge_vec,
    ):
        attn_act = act_fn_map[self.attn_activation]
        attn = (q_i * k_j * dk).sum(axis=-1)

        attn = attn_act(attn) * self.cutoff_fn(edge_weight)

        v_j = v_j * dv
        x, vec1, vec2 = jnp.split(v_j, 3, axis=2)

        x = x * attn[:, :, jnp.newaxis]

        vec = (
            vec_j * vec1[:, jnp.newaxis]
            + vec2[:, jnp.newaxis] * edge_vec[:, :, jnp.newaxis, jnp.newaxis]
        )

        return x, vec

    def _aggregate(self, features, index, num_segments):
        x, vec = features
        x = jax.ops.segment_sum(x, segment_ids=index, num_segments=num_segments)
        vec = jax.ops.segment_sum(vec, segment_ids=index, num_segments=num_segments)

        return x, vec

    def _propagate(
        self,
        senders,
        receivers,
        q,
        k,
        v,
        vec,
        dk,
        dv,
        edge_weight,
        edge_vec,
        num_segments,
    ):
        q_i = q[receivers]
        k_j = k[senders]
        v_j = v[senders]
        vec_j = vec[senders]
        x, vec = self._message(
            q_i,
            k_j,
            v_j,
            vec_j,
            dk,
            dv,
            edge_weight,
            edge_vec,
        )

        return self._aggregate((x, vec), receivers, num_segments)

    @nn.compact
    def __call__(self, x, vec, senders, receivers, edge_weight, edge_attr, edge_vec):
        dense_w_init = Partial(
            nn.Dense,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.constant(0.0),
        )

        x = nn.LayerNorm()(x)

        q = dense_w_init(self.hidden_channels)(x).reshape(
            -1, self.num_heads, self.head_dim
        )
        k = dense_w_init(self.hidden_channels)(x).reshape(
            -1, self.num_heads, self.head_dim
        )

        if self.qk_norm:
            q = nn.LayerNorm()(q)
            k = nn.LayerNorm()(k)

        v = dense_w_init(self.hidden_channels * 3)(x).reshape(
            -1, self.num_heads, self.head_dim * 3
        )

        vec1, vec2, vec3 = jnp.split(
            dense_w_init(self.hidden_channels * 3, use_bias=False)(vec), 3, axis=-1
        )

        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(axis=1)

        act_fn = act_fn_map[self.activation]
        dk = act_fn(dense_w_init(self.hidden_channels)(edge_attr)).reshape(
            -1, self.num_heads, self.head_dim
        )

        dv = act_fn(dense_w_init(self.hidden_channels * 3)(edge_attr)).reshape(
            -1, self.num_heads, self.head_dim * 3
        )

        x, vec = self._propagate(
            senders,
            receivers,
            q,
            k,
            v,
            vec,
            dk,
            dv,
            edge_weight,
            edge_vec,
            num_segments=x.shape[0],
        )

        # print(x.shape)
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        if self.norm_coors:
            vec = CoorsNorm(scale_init=self.norm_coors_scale_init)(vec)

        o1, o2, o3 = jnp.split(dense_w_init(self.hidden_channels * 3)(x), 3, axis=1)

        # print(o1.shape)
        dvec = vec3 * o1[:, jnp.newaxis] + vec
        dx = vec_dot * o2 + o3
        return dx, dvec
