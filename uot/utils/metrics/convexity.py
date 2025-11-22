from collections.abc import Sequence

import jax
import jax.numpy as jnp

def compute_convexity_and_condition(T: jnp.ndarray, grid_spacings: Sequence[float], tol=1e-10, eps=1e-12) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    T: array of shape (n1, n2, ..., nd, d) representing Monge map
    grid_spacings: tuple of length d giving spacing h_k along each axis
    Returns:
    is_convex: boolean mask same spatial shape, True if convex locally
    min_eig: array of local minimum eigenvalues
    cond_num: array of local condition numbers
    """
    *grid_shape, d = T.shape

    def jacobian_at_point(idx):
        J = jnp.zeros((d, d))
        for beta in range(d):
            idx_f = list(idx)
            idx_b = list(idx)
            idx_f[beta] += 1
            idx_b[beta] -= 1
            idx_f[beta] = jnp.clip(idx_f[beta], 0, grid_shape[beta] - 1)
            idx_b[beta] = jnp.clip(idx_b[beta], 0, grid_shape[beta] - 1)
            T_f = T[tuple(idx_f)]
            T_b = T[tuple(idx_b)]
            diff = (T_f - T_b) / (2 * grid_spacings[beta])
            J = J.at[:, beta].set(diff)
        return 0.5 * (J + J.T)

    all_indices = jnp.stack(jnp.meshgrid(
        *[jnp.arange(n) for n in grid_shape], indexing='ij'
    ), axis=-1).reshape(-1, d)
    
    def process_idx(idx):
        H = jacobian_at_point(idx)
        eigs = jnp.linalg.eigvalsh(H)
        lam_min = eigs[0]
        lam_max = eigs[-1]
        is_cvx = lam_min >= -tol
        cond = lam_max / jnp.maximum(lam_min, eps)
        return is_cvx, lam_min, cond

    is_cvx, lam_min, cond_num = jax.vmap(process_idx)(all_indices)
    is_cvx = is_cvx.reshape(tuple(grid_shape))
    lam_min = lam_min.reshape(tuple(grid_shape))
    cond_num = cond_num.reshape(tuple(grid_shape))
    return is_cvx, lam_min, cond_num