from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike

from uot.utils.solver_helpers import tensor_marginals, coupling_tensor

from ott.solvers import linear
from ott.geometry import geometry


class OTTSinkhornSolver(BaseSolver):
    def __init__(self):
        return super().__init__()

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        reg: float = 1e-3,
        maxiter: int = 1000,
        tol: float = 1e-6,
    ) -> dict:
        if len(marginals) != 2:
            raise ValueError("Sinkhorn solver accepts only two marginals.")
        if len(costs) == 0:
            raise ValueError("Cost tensors not defined.")
        mu, nu = marginals[0], marginals[1]

        c = costs[0]
        geom = geometry.Geometry(cost_matrix=c, epsilon=reg)

        solve_jit = jax.jit(
            linear.solve,
            static_argnames=('max_iterations', 'threshold', 'lse_mode')
        )

        out = solve_jit(geom, a=mu.to_discrete()[1], b=nu.to_discrete()[1],
                     max_iterations=maxiter, threshold=tol, lse_mode=True)

        transport_plan = out.matrix

        return {
            "transport_plan": transport_plan ,
            "cost": (transport_plan * c).sum(),
            "u_final": out.f,
            "v_final": out.g,
            "iterations": out.n_iters,
            "error": jnp.max(out.errors),
        }
