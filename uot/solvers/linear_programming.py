import ot
import numpy as np
from typing import Sequence

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver

from uot.utils.types import ArrayLike

class LinearProgrammingTwoMarginalSolver(BaseSolver):
    
    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
    ) -> dict:
        if len(costs) == 0:
            raise ValueError("Cost tensors not defined.")
        if len(marginals) != 2:
            raise ValueError("This linear programming solver accepts only two marginals.")
        mu, nu = marginals[0].to_discrete()[1], marginals[1].to_discrete()[1]

        mu = np.asarray(mu)
        nu = np.asarray(nu)
        cost = np.asarray(costs[0])

        coupling, log = ot.emd(mu, nu, cost, log=True)

        return {
            "transport_plan": coupling,
            "cost": log['cost'],
            "u_final": log['u'],
            "v_final": log['v'],
            "iterations": None,
            "error": 0,
        }
