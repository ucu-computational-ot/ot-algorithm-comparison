from uot.utils.types import ArrayLike
import numpy as np
import jax.numpy as jnp


def generate_nd_grid(axes: list[ArrayLike], use_jax: bool = True) -> ArrayLike:
    """
    Given a list of 1D arrays (or array‐like) `axes`, each of length m_i,
    return an array of shape (∏ m_i, n) whose rows are all points in the
    Cartesian product grid.
    If use_jax=True, runs under jax.numpy.

    Args:
        axes: list of length n, where each element is a 1D sequence or ndarray
              representing the grid values along that dimension.

    Returns:
        points: np.ndarray of shape (M, n), where M = product of len(axes[i]).
                Each row is an n‐tuple (x1, x2, ..., xn) corresponding to one
                grid point.
    """
    # ——— Handle empty axes list ———
    if len(axes) == 0:
        if use_jax:
            return jnp.zeros((1, 0), dtype=jnp.float32)
        return np.zeros((1, 0), dtype=float)
    xp = jnp if use_jax else np
    grid_axes = [xp.asarray(a).ravel() for a in axes]
    mesh = xp.meshgrid(*grid_axes, indexing='ij')
    stacked = xp.stack(mesh, axis=-1)  # shape (m_0, m_1, …, m_{n-1}, n)
    points = stacked.reshape(-1, len(grid_axes))
    return points
