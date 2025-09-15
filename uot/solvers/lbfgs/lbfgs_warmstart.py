from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jaxopt import LBFGS, OptStep

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike
from uot.utils.solver_helpers import coupling_tensor

# Import your existing Sinkhorn initialization function
# (_sinkhorn or sinkhorn_jax) from uot.solvers.sinkhorn
from uot.solvers.sinkhorn import _sinkhorn  # or sinkhorn_jax


class WarmStartLBFGSTwoMarginalSolver(BaseSolver):
    def __init__(self, history_size: int = 10, line_search: str = "zoom", maxls: int = 100):
        super().__init__()
        self.history_size = history_size
        self.line_search = line_search
        self.maxls = maxls

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        reg: float = 1e-3,
        maxiter: int = 1000,
        tol: float = 1e-6,
    ) -> dict:
        if len(marginals) != 2:
            raise ValueError("WarmStartLBFGS solver accepts only two marginals.")
        if len(costs) == 0:
            raise ValueError("Cost matrix not provided.")

        mu, nu = marginals[0].to_discrete()[1], marginals[1].to_discrete()[1]
        C = costs[0]

        # warm start with Sinkhorn to get initial u0, v0
        u0, v0, _, _ = _sinkhorn(
            a=mu,
            b=nu,
            cost=C,
            epsilon=reg,
            precision=tol * 10,       # looser tol for speed
            max_iters=100              # few iterations
        )

        def dual_objective(uv):
            u, v = uv
            logK = (u[:, None] + v[None, :] - C) / reg
            m = jnp.max(logK, axis=None, keepdims=False)
            ent = reg * jnp.sum(jnp.exp(logK - m)) * jnp.exp(m)
            return -(jnp.dot(u, mu) + jnp.dot(v, nu) - ent)

        solver = LBFGS(
            fun=dual_objective,
            tol=tol,
            maxiter=maxiter,
            history_size=self.history_size,
            linesearch=self.line_search,
            maxls=self.maxls,
            implicit_diff=False,
        )

        init_params = (u0, v0)
        result: OptStep = solver.run(init_params)

        u_final, v_final = result.params
        plan = coupling_tensor(u_final[None, :], v_final[:, None], C, reg)
        cost = jnp.sum(plan * C)

        row_err = jnp.max(jnp.abs(plan.sum(axis=1) - mu))
        col_err = jnp.max(jnp.abs(plan.sum(axis=0) - nu))
        final_err = jnp.maximum(row_err, col_err)

        return {
            "transport_plan": plan,
            "cost": cost,
            "u_final": u_final,
            "v_final": v_final,
            "iterations": int(result.state.iter_num),
            "error": float(final_err),
        }
