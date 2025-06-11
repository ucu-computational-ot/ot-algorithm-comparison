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
        # TODO: implement this GridMeasure class

    def to_discrete(self):
        return super().to_discrete()
