from abc import ABC
import numpy as np
from typing import List, Callable
from uot.data.measure import BaseMeasure
from uot.utils.types import ArrayLike


class MarginalProblem(ABC):
    def __init__(
        self, name: str, measures: List[BaseMeasure], cost_fns: List[Callable]
    ):
        super().__init__()
        if len(measures) < 2:
            raise ValueError("Need at least two marginals")
        self.name = name
        self.measures = measures
        self.cost_fns = cost_fns
        # guys, I followed your approach to cache the cost function
        # BUT I think that there might be better ones: just store all cost function or use other caching procedures. so I ask you:
        # TODO: for now just compute WHOLE cost matrix and store it as it is.
        self._cost_cache = [None] * len(cost_fns)

    def get_marginals(self) -> List[BaseMeasure]:
        raise NotImplementedError()

    def get_costs(self) -> List[ArrayLike]:
        raise NotImplementedError()

    def to_dict(self) -> dict:
        raise NotImplementedError()

    def free_memory(self):
        # TODO: do we actually need this one?
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.name}"