import numpy as np
import jax.numpy as jnp
from uot.utils.types import ArrayLike


def get_axes(dim: int,
             borders: tuple[float, float],
             n_points: int,
             use_jax: bool = True) -> ArrayLike:
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
    lib = jnp if use_jax else np
    ax = lib.linspace(borders[0], borders[1], n_points)
    axs = [ax for _ in range(dim)]
    return axs
