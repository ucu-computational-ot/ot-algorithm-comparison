from abc import ABC
import hashlib
import pickle
from collections.abc import Callable
from uot.data.measure import BaseMeasure
from uot.utils.types import ArrayLike


class MarginalProblem(ABC):
    def __init__(
        self, name: str, measures: list[BaseMeasure], cost_fns: list[Callable]
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
        self.__hash = None

    def __repr__(self):
        space_size = 'x'.join(
            str(marginal.to_discrete()[0].size)
            for marginal in self.get_marginals()
        )
        return f"<{self.__class__.__name__}[{self.name}] {space_size}\
        with ({map(lambda fn: fn.__name__, self.cost_fns)})>"

    def key(self) -> str:
        if self.__hash is None:
            blob = pickle.dumps(self, protocol=4)
            self.__hash = hashlib.sha1(blob).hexdigest()
        return self.__hash

    def __hash__(self) -> int:
        """
        Return an integer hash, derived from the SHA1 key.
        """
        # Convert the first 16 hex digits of the key into a Python int
        hex_key = self.key()[:16]
        return int(hex_key, 16)

    def get_marginals(self) -> list[BaseMeasure]:
        raise NotImplementedError()

    def get_costs(self) -> list[ArrayLike]:
        raise NotImplementedError()

    def to_dict(self) -> dict:
        raise NotImplementedError()

    def free_memory(self):
        # TODO: do we actually need this one?
        raise NotImplementedError()
