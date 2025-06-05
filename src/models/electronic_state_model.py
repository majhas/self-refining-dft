import jax.numpy as jnp
from flax import linen as nn

from src.dft.hamiltonian import Hamiltonian
from src.dft.property import total_energy
from src.models.components.coefficient_net import CoefficientNetwork
from src.networks.etnn.model import EquivariantTransformer

activation_mapping = {"silu": nn.silu, "swish": nn.swish}


class ElectronicStateModel(nn.Module):
    # Equivariant Transformer network args
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

    # Coefficient network args
    num_coef_layers: int = 4
    num_coef_heads: int = 8
    max_orbital_species: int = 20
    norm_coefficients: bool = False
    density_mixing: bool = False
    resnet: bool = False
    use_v2: bool = False
    default_bias: bool = True

    @nn.compact
    def __call__(
        self,
        atomic_number: jnp.ndarray,
        position: jnp.ndarray,
        orbital_tokens,
        orbital_index,
        senders,
        receivers,
        hamiltonian: Hamiltonian,
        output_coefficients: bool = False,
    ):
        x, _ = EquivariantTransformer(
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

        C = CoefficientNetwork(
            hidden_channels=self.hidden_channels,
            num_layers=self.num_coef_layers,
            num_heads=self.num_coef_heads,
            max_species=self.max_orbital_species,
            activation=self.activation,
            norm_coefficients=self.norm_coefficients,
            resnet=self.resnet,
            density_mixing=self.density_mixing,
            default_bias=self.default_bias,
        )(
            x=x,
            orbital_tokens=orbital_tokens,
            orbital_index=orbital_index,
            hamiltonian=hamiltonian,
        )

        # sum of electronic energy and nuclear energy
        E_total = total_energy(
            H=hamiltonian, C=C, atomic_number=atomic_number, position=position
        )

        if output_coefficients:
            return E_total, C

        # only electronic energy is needed for updates wrt to params
        # but need the total energy for sampling with langevin (grad energy wrt to positions)
        return E_total
