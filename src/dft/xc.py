import jax
import jax.numpy as jnp

CLIP_RHO_MIN = 1e-10
CLIP_RHO_MAX = 1e15


def lyp(n, gnn):
    # precompute
    A = 0.04918
    B = 0.132
    C = 0.2533
    Dd = 0.349
    CF = 0.3 * (3.0 * jnp.pi * jnp.pi) ** (2.0 / 3.0)
    c0 = 2.0 ** (11.0 / 3.0) * (1 / 2) ** (8 / 3)
    c1 = (1 / 3 + 1 / 8) * 4

    # actual compute
    log_n = jnp.log(n)
    icbrtn = jnp.exp(log_n * (-1.0 / 3.0))

    P = 1.0 / (1.0 + Dd * icbrtn)
    omega = jnp.exp(-C * icbrtn) * P
    delta = icbrtn * (C + Dd * P)

    n_five_three = jnp.exp(log_n * (-5 / 3))

    result = -A * (
        n * P
        + B
        * omega
        * 1
        / 4
        * (
            2 * CF * n * c0
            + gnn * (60 - 14.0 * delta) / 36 * n_five_three
            - gnn * c1 * n_five_three
        )
    )

    return result


def lda(rho):
    return -jnp.exp(1 / 3 * jnp.log(rho) - 0.30305460484554375)


def b88(a, gaa):
    # precompute
    c1 = 4.0 / 3.0
    c2 = -8.0 / 3.0
    c3 = (-3.0 / 4.0) * (6.0 / jnp.pi) ** (1.0 / 3.0) * 2
    d = 0.0042

    # actual compute
    log_a = jnp.log(a / 2)
    na43 = jnp.exp(log_a * c1)
    chi2 = gaa / 4 * jnp.exp(log_a * c2)
    chi = jnp.exp(jnp.log(chi2) / 2)
    b88 = -(d * na43 * chi2) / (1.0 + 6 * d * chi * jnp.arcsinh(chi)) * 2
    slaterx_a = c3 * na43
    return slaterx_a + b88


def vwn(n):
    # Precompute stuff in np.float64
    p = jnp.array([-0.10498, 0.0621813817393097900698817274255, 3.72744, 12.9352])
    f = p[0] * p[2] / (p[0] * p[0] + p[0] * p[2] + p[3]) - 1.0
    f_inv_p1 = 1 / f + 1
    f_2 = f * 0.5
    sqrt = jnp.sqrt(4.0 * p[3] - p[2] * p[2])
    precompute = p[2] * (
        1.0 / sqrt
        - p[0] / ((p[0] * p[0] + p[0] * p[2] + p[3]) * sqrt / (p[2] + 2.0 * p[0]))
    )
    log_s_c = jnp.log(3.0 / (4 * jnp.pi)) / 6

    # Below cast to same dtype as input (allow easier comparison between f32/f64).
    dtype = n.dtype
    p = p.astype(dtype)
    f = f.astype(dtype)
    f_inv_p1 = (f_inv_p1).astype(dtype)
    f_2 = f_2.astype(dtype)
    sqrt = sqrt.astype(dtype)
    precompute = precompute.astype(dtype)
    log_s_c = log_s_c.astype(dtype)

    # compute stuff that depends on n
    log_s = -jnp.log(n) / 6 + log_s_c
    s_2 = jnp.exp(log_s * 2)
    s = jnp.exp(log_s)
    z = sqrt / (2.0 * s + p[2])

    result = (
        n
        * p[1]
        * (
            log_s
            + f * jnp.log(jnp.sqrt(s_2 + p[2] * s + p[3]) / (s - p[0]) ** (f_inv_p1))
            + precompute * jnp.arctan(z)
        )
    )

    return result


def b3lyp(rho, EPSILON_B3LYP=0):
    rho = jnp.concatenate([jnp.clip(rho[:1], CLIP_RHO_MIN, CLIP_RHO_MAX), rho[1:4] * 2])

    rho0 = rho.T[:, 0]
    norms = jnp.linalg.norm(rho[1:], axis=0).T ** 2 + EPSILON_B3LYP

    def _lda(rho0):
        return jax.vmap(jax.value_and_grad(lambda x: lda(x) * 0.08))(rho0)

    def _vwn(rho0):
        return jax.vmap(jax.value_and_grad(lambda x: vwn(x) * 0.19))(rho0)

    def _b88(rho0, norms):
        return jax.vmap(
            jax.value_and_grad(lambda rho0, norm: b88(rho0, norm) * 0.72, (0, 1))
        )(rho0, norms)

    def _lyp(rho0, norms):
        return jax.vmap(
            jax.value_and_grad(lambda rho0, norm: lyp(rho0, norm) * 0.810, (0, 1))
        )(rho0, norms)

    e_xc_lda, v_rho_lda = jax.jit(_lda)(rho0)
    e_xc_vwn, v_rho_vwn = jax.jit(_vwn)(rho0)
    e_xc_b88, (v_rho_b88, v_norm_b88) = jax.jit(_b88)(rho0, norms)
    e_xc_lyp, (v_rho_lyp, v_norm_lyp) = jax.jit(_lyp)(rho0, norms)

    e_xc = e_xc_lda + (e_xc_vwn + e_xc_b88 + e_xc_lyp) / rho0
    v_xc_rho = v_rho_lda * 4 * rho0 + v_rho_vwn + v_rho_b88 + v_rho_lyp
    v_xc_norms = v_norm_b88 + v_norm_lyp

    return e_xc, v_xc_rho, v_xc_norms
