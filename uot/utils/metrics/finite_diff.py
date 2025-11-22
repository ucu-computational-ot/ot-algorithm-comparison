import jax
import jax.numpy as jnp
from uot.utils.central_gradient_nd import _central_gradient_nd
from typing import Sequence


def hessian_via_fd(psi: jnp.ndarray) -> jnp.ndarray:
    """
    Build Hessian by reusing central-diff gradient:
      H[i,j,...] = ∂^2 psi / ∂x_i ∂x_j
    Same [0,1]^d, h_i = 1/n_i convention as _central_gradient_nd.
    """
    g = _central_gradient_nd(psi)                         # (d, *shape)
    H_list = [_central_gradient_nd(g[i]) for i in range(g.shape[0])]
    H = jnp.stack(H_list, axis=0)                         # (d, d, *shape)
    return H

def compute_jacobian_nd(T: jnp.ndarray, grid_spacings: Sequence[float]) -> jnp.ndarray:
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
        return J  # full Jacobian, not symmetrized

    all_indices = jnp.stack(jnp.meshgrid(
        *[jnp.arange(n) for n in grid_shape], indexing='ij'
    ), axis=-1).reshape(-1, d)

    all_J = jax.vmap(jacobian_at_point)(all_indices)  # (num_points, d, d)
    all_J = all_J.reshape(tuple(grid_shape) + (d, d))
    all_J = jnp.moveaxis(all_J, (-2, -1), (0, 1))  # (d, d, *grid_shape)
    return all_J