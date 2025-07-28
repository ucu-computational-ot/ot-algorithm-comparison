import ot
from collections.abc import Callable
from uot.data.measure import BaseMeasure
from uot.problems.base_problem import MarginalProblem
from uot.utils.types import ArrayLike

from uot.utils.logging import logger


class BarycenterProblem(MarginalProblem):
    def __init__(
        self,
        name: str,
        measures: list[BaseMeasure],
        weights: ArrayLike,
        cost_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    ):
        super().__init__(name, measures, [cost_fn])
        self._measures = [measure.get_jax() for measure in measures]
        self._cost_fn = cost_fn
        self._C = None
        self._weights = weights

        self._exact_cost = None
        self._exact_coupling = None

    def get_marginals(self) -> list[BaseMeasure]:
        return self._measures

    def get_costs(self) -> list[ArrayLike]:
        """
        Returns a single‐element list containing the cost matrix between
        self._mu and self._nu, caching it in self._C so that repeated
        calls do not recompute.
        """
        if self._C is None:
            X, _ = self._measures[0].to_discrete()  # X: ArrayLike of shape (n, d)

            C = self._cost_fn(X, X)  # should return an (n × m) array

            self._C = [C]

        return self._C

    def to_dict(self) -> dict:
        size = len(self.measures[0].to_discrete()[0])
        return {
            "dataset": self.name,
            "size": size,
            "cost": self._cost_fn.__name__,
        }