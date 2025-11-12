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


def compute_axis_spacings(axes: list[ArrayLike], use_jax: bool = True) -> list[ArrayLike]:
    """
    Return per-axis spacings assuming uniform grids (cell-centered or vertex-centered).
    For singleton axes, fall back to spacing 1.0 to avoid division-by-zero downstream.
    """
    xp = jnp if use_jax else np
    spacings: list[ArrayLike] = []
    for i, axis in enumerate(axes):
        arr = xp.asarray(axis)
        if arr.ndim != 1:
            raise ValueError(f"Axis {i} must be 1-D, got shape {arr.shape}")
        if arr.size <= 1:
            spacing = xp.asarray(1.0) if use_jax else 1.0
        else:
            spacing = arr[1] - arr[0]
        spacings.append(spacing)
    return spacings


def compute_cell_volume(axes: list[ArrayLike], use_jax: bool = True) -> float:
    """
    Compute the hyper-rectangular cell volume implied by the provided axes.
    Useful when approximating integrals via cell-centered sampling.
    """
    spacings = compute_axis_spacings(axes, use_jax=use_jax)
    volume = 1.0
    for spacing in spacings:
        volume *= float(spacing)
    return volume
