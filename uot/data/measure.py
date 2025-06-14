import numpy as np
from abc import ABC, abstractmethod

from uot.utils.types import ArrayLike


class BaseMeasure(ABC):
    @abstractmethod
    def to_discrete(self) -> tuple[ArrayLike, ArrayLike]:
        "Return (points, weights) that approximate this measure as the discrete one"
        pass


class DiscreteMeasure(BaseMeasure):
    def __init__(self, points: ArrayLike, weights: ArrayLike, name: str = ""):
        super().__init__()
        self._points = points
        self._weights = weights
        self.name = name

    def to_discrete(self):
        return self._points, self._weights


class GridMeasure(BaseMeasure):
    def __init__(self,
                 axes: list[np.ndarray],
                 weights_nd: np.ndarray,
                 name: str = "",
                 normalize: bool = True,
                 ):
        super().__init__()

        if len(axes) != weights_nd.ndim:
            raise ValueError(
                f"Number of axes ({len(axes)}) must match the number of dimensions in weights_nd ({weights_nd.ndim})"
            )

        for i, axis in enumerate(axes):
            if axis.shape[0] != weights_nd.shape[i]:
                raise ValueError(
                f"Axis {i} length ({axis.shape[0]}) does not match weights dimension ({weights_nd.shape[i]})"
            )

        self._axes = axes
        self._weights_nd = weights_nd
        self.name = name

        if normalize:
            total_mass = np.sum(weights_nd)
            if total_mass > 0:
                self._weights_nd = weights_nd / total_mass

    def to_discrete(self):
        mesh = np.meshgrid(*self._axes, indexing='ij')

        points = np.stack([m.ravel() for m in mesh], axis=-1)
        weights = self._weights_nd.ravel()

        non_zero = weights > 0
        return points[non_zero], weights[non_zero]
