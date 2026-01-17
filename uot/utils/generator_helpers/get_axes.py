from typing import Literal
import numpy as np
import jax.numpy as jnp
from uot.utils.types import ArrayLike

CellDiscretization = Literal[
    'cell-centered', 'vertex-centered'
]


def cell_centered_axes(n_points: int, L0: float, L1: float, use_jax: bool = True):
    lib = jnp if use_jax else np
    length = L1 - L0
    h = length / (n_points)
    ax = lib.linspace(L0+0.5*h, L1-0.5*h, n_points)
    return ax


def vertex_centered_axes(n_points: int, L0: float, L1: float, use_jax: bool = True):
    lib = jnp if use_jax else np
    ax = lib.linspace(L0, L1, n_points)
    return ax


def get_axes(dim: int,
             borders: tuple[float, float],
             n_points: int,
             cell_discretization: CellDiscretization = 'cell-centered',
             use_jax: bool = True,
             ) -> list[ArrayLike]:
    """
    Generate a list of 1D coordinate axes for an n-dimensional grid.

    Each axis is a linearly spaced array of length `n_points` between
    `borders[0]` and `borders[1]`. Returns the same axis array repeated
    `dim` times.

    Parameters
    ----------
    dim : int
        Number of dimensions (i.e., how many axes to return).
    borders : tuple of float
        Two‐tuple `(lower, upper)` specifying the interval for the axes.
    n_points : int
        Number of points on each axis.
    use_jax : bool, optional
        If True, use `jax.numpy.linspace` to build the axes; otherwise use
        `numpy.linspace`. Default is True.

    Returns
    -------
    List[array_like]
        A list of length `dim`, where each element is a 1D array (NumPy or
        JAX array) of shape `(n_points,)` containing the linearly spaced values.
    """
    if cell_discretization == 'cell-centered':
        ax = cell_centered_axes(n_points, borders[0], borders[1], use_jax)
    elif cell_discretization == 'vertex-centered':
        ax = vertex_centered_axes(n_points, borders[0], borders[1], use_jax)
    axs = [ax for _ in range(dim)]
    return axs
