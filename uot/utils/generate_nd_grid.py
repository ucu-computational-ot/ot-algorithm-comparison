from uot.utils.types import ArrayLike
import numpy as np
from typing import List


def generate_nd_grid(axes: List[ArrayLike]) -> np.ndarray:
    """
    Given a list of 1D arrays (or array‐like) `axes`, each of length m_i,
    return an array of shape (∏ m_i, n) whose rows are all points in the
    Cartesian product grid.

    Args:
        axes: list of length n, where each element is a 1D sequence or ndarray
              representing the grid values along that dimension.

    Returns:
        points: np.ndarray of shape (M, n), where M = product of len(axes[i]).
                Each row is an n‐tuple (x1, x2, ..., xn) corresponding to one
                grid point.
    """
    # Convert each axis to a NumPy array
    grid_axes = [np.asarray(a).ravel() for a in axes]
    n = len(grid_axes)

    # Use np.meshgrid with indexing='ij' to create an n‐dimensional mesh
    mesh = np.meshgrid(*grid_axes, indexing='ij')

    # Each element of mesh is an array of shape (m_0, m_1, ..., m_{n-1})
    # Stack them along a new last axis to get shape (…, n)
    stacked = np.stack(mesh, axis=-1)  # shape (m_0, m_1, …, m_{n-1}, n)

    # Flatten the first n dimensions to a single axis of length M = ∏ m_i
    M = stacked.shape[:-1]
    num_points = np.prod(M)
    points = stacked.reshape(num_points, n)

    return points
