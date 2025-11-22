import jax

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike

from jax import numpy as jnp
from uot.algorithms.rapdhg import raPDHG, create_ot_problem
from uot.algorithms.rapdhg.strategies.reg_strategy import RegStrategy
from uot.algorithms.rapdhg.utils import RestartScheme
from uot.algorithms.rapdhg.utils import OTProblem



from typing import Sequence


class PDLPSolver(BaseSolver):
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
            raise ValueError("PDLP solver accepts only two marginals.")
        if len(costs) == 0:
            raise ValueError("Cost tensors not defined.")
        # mu, nu = marginals[0], marginals[1]
        mu, nu = (
            marginals[0].to_discrete(include_zeros=True)[1],
            marginals[1].to_discrete(include_zeros=True)[1],
        )
        coupling, u, v, i_final, final_err = _solve_pdlp(
            a=mu,
            b=nu,
            cost=costs[0],
            epsilon=reg,
            precision=tol,
            max_iters=maxiter,
        )
        return {
            "transport_plan": coupling,
            "u_final": u,
            "v_final": v,
            "cost": jnp.sum(costs[0] * coupling),
            "iterations": i_final,
            "error": final_err,
        }

@jax.jit
def _solve_pdlp(
    a: jnp.ndarray,
    b: jnp.ndarray,
    cost: jnp.ndarray,
    epsilon: float = 1e-3,
    precision: float = 1e-4,
    max_iters: int = 10_000,
):
    solver = raPDHG(
        verbose=False,
        jit=True,
        reg=epsilon,
        eps_abs=precision,
        eps_rel=precision,
        iteration_limit=max_iters,
        termination_evaluation_frequency=40,
    )
    problem = create_ot_problem(cost, a, b)
    result, ci = solver.optimize(problem, dim=problem.n)

    coupling = result.avg_primal_solution
    u = result.avg_dual_solution[0]
    v = result.avg_dual_solution[1]
    iters = result.num_iterations
    error = ci.primal_residual_norm

    return coupling, u, v, iters, error
