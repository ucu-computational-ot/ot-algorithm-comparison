import ot
from collections.abc import Callable
from uot.data.measure import BaseMeasure
from uot.problems.base_problem import MarginalProblem
from uot.utils.types import ArrayLike

from uot.utils.logging import logger


class TwoMarginalProblem(MarginalProblem):
    def __init__(
        self,
        name: str,
        mu: BaseMeasure,
        nu: BaseMeasure,
        cost_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    ):
        super().__init__(name, [mu, nu], [cost_fn])
        self._mu = mu.get_jax()
        self._nu = nu.get_jax()
        self._cost_fn = cost_fn
        self._C = None

        self._exact_cost = None
        self._exact_coupling = None

    def get_marginals(self) -> list[BaseMeasure]:
        return [self._mu, self._nu]

    def get_costs(self) -> list[ArrayLike]:
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

    def get_exact_cost(self) -> float:
        """
        Return exact cost of transportation between measures
        self._mu and self._nu, caching it in the self._exact_cost,
        such that repeated calls do not recompute
        """
        if self._exact_cost is None:
            self._compute_exact_solution()
        return self._exact_cost

    def get_exact_coupling(self) -> float:
        """
        Return exact map of transportation between measures
        self._mu and self._nu, caching it in the self._exact_cost,
        such that repeated calls do not recompute
        """
        if self._exact_coupling is None:
            self._compute_exact_solution()
        return self._exact_coupling

    def to_dict(self) -> dict:
        mu_size = len(self._mu.to_discrete()[0])
        nu_size = len(self._nu.to_discrete()[0])
        return {
            "dataset": self.name,
            "mu_size": mu_size,
            "nu_size": nu_size,
            "cost": self._cost_fn.__name__,
        }

    def _compute_exact_solution(self):
        """
        Compute exact solution of transportation between
        self._mu and self._nu and cache it in self._exact_cost and 
        self._exact_coupling
        """
        a = self._mu.to_discrete()[1]
        b = self._nu.to_discrete()[1]
        C = self.get_costs()[0]

        T, log = ot.emd(a, b, C, log=True, numItermax=10000000)
        if log['warning'] is not None:
            logger.warning(f"Computing ground truth for {self.to_dict()} didn't converge")

        self._exact_coupling = T
        self._exact_cost = log['cost']

    def free_memory(self):
        # TODO: as mentioned in the abstract class, consider removing this
        #       method, as we should move all the responsiblity of the memory
        #       management on the GC
        self._C = None
