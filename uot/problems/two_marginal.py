from typing import Callable, List
from uot.data.measure import BaseMeasure
from uot.problems.base_problem import MarginalProblem
from uot.utils.types import ArrayLike


class TwoMarginalProblem(MarginalProblem):
    def __init__(
        self,
        name: str,
        mu: BaseMeasure,
        nu: BaseMeasure,
        cost_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    ):
        super().__init__(name, [mu, nu], [cost_fn])
        self._mu = mu
        self._nu = nu
        self._cost_fn = cost_fn
        self._C = None

    def get_marginals(self) -> List[BaseMeasure]:
        return [self._mu, self._nu]

    def get_costs(self) -> List[ArrayLike]:
        """
        Returns a single‐element list containing the cost matrix between
        self._mu and self._nu, caching it in self._C so that repeated
        calls do not recompute.
        """
        if self._C is None:
            X, _ = self._mu.to_discrete()  # X: ArrayLike of shape (n, d)
            Y, _ = self._nu.to_discrete()  # Y: ArrayLike of shape (m, d)

            C = self._cost_fn(X, Y)  # should return an (n × m) array

            self._C = [C]

        return self._C

    def to_dict(self) -> dict:
        mu_size = len(self._mu.to_discrete()[0])
        nu_size = len(self._nu.to_discrete()[0])
        return {
            "dataset": self.name,
            "mu_size": mu_size,
            "nu_size": nu_size,
            "cost": self._cost_fn.__name__,
        }

    def free_memory(self):
        # TODO: as mentioned in the abstract class, consider removing this
        #       method, as we should move all the responsiblity of the memory
        #       management on the GC
        self._C = None
