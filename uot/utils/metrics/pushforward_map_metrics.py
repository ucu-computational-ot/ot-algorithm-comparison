from collections.abc import Sequence

import jax.numpy as jnp

from .finite_diff import compute_jacobian_nd
from .interpolation import linear_sample_nd
from .convexity import compute_convexity_and_condition

from uot.utils.types import ArrayLike

def extra_grid_metrics(
    *,
    mu_nd: jnp.ndarray,
    nu_nd: jnp.ndarray,
    axes_mu: Sequence[ArrayLike],
    X: jnp.ndarray,
    T: jnp.ndarray,
    pushforward_mu: jnp.ndarray = None,
) -> dict:
    """
    Computes:
      - TV distance between pushforward of mu and nu
        - `tv_mu_to_nu`
      - Monge–Ampère residual norms
        - `ma_residual_L1`, `ma_residual_Linf`
      - det(J) diagnostics

    NOTE on convention:
      We use general T(x) for metrics, assuming quadratic cost OT map.
    """
    hs = [ax[1] - ax[0] if ax.shape[0] > 1 else 1.0 for ax in axes_mu]

    # (A) Push-forward TV distance
    tv_mu_to_nu = 0.5 * jnp.sum(jnp.abs(pushforward_mu - nu_nd))

    # (B) Monge–Ampère residual
    J = compute_jacobian_nd(T, hs)  # (d, d, *shape)
    d = mu_nd.ndim
    shape = mu_nd.shape
    detJ = jnp.linalg.det(J.reshape(d, d, -1).transpose(2, 0, 1)).reshape(shape)

    # sample rho_nu at mapped points T(x)
    rho_nu_at_T = linear_sample_nd(nu_nd, T, axes_mu)

    ma_residual = rho_nu_at_T * detJ - mu_nd
    ma_L1 = jnp.sum(jnp.abs(ma_residual))
    ma_Linf = jnp.max(jnp.abs(ma_residual))

    # det(J) sanity stats
    detJ_min = jnp.min(detJ)
    detJ_max = jnp.max(detJ)
    detJ_neg_frac = jnp.mean((detJ < 0).astype(jnp.float32))

    # compute the convexity of the potential
    (is_convex, eigenvalues, condition_number) = compute_convexity_and_condition(T, hs)
    convexity_fraction = jnp.mean(is_convex)
    eigenvalue_min = jnp.min(eigenvalues)
    condition_number_max = jnp.max(condition_number)

    return {
        "tv_mu_to_nu": tv_mu_to_nu,
        "ma_residual_L1": ma_L1,
        "ma_residual_Linf": ma_Linf,
        "detJ_min": detJ_min,
        "detJ_max": detJ_max,
        "detJ_neg_frac": detJ_neg_frac,
        "phi_is_convex": convexity_fraction,
        "eigenvalue_min_phi_hessian": eigenvalue_min,
        "condition_number_hessian_eigenvalues": condition_number_max,
    }