import jax

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike
from uot.utils.logging import logger

from jax import numpy as jnp
from uot.algorithms.pdlp_bary import raPDHG, create_barycenter_problem
from functools import partial

from typing import Sequence


class PDLPBarycenterSolver(BaseSolver):
    def __init__(self):
        return super().__init__()

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        weights: ArrayLike,
        reg: float = 1e-3,
        maxiter: int = 1000,
        tol: float = 1e-6,
    ) -> dict:
        if len(costs) == 0:
            raise ValueError("Cost tensors not defined.")

        couplings, barycenter, us, vs, i_final, final_err = _solve_pdlp_barycenter(
            cost=costs[0],
            marginals = jnp.asarray([marg.to_discrete()[1] for marg in marginals]),
            weights = jnp.asarray(weights),
            epsilon=reg,
            precision=tol,
            max_iters=maxiter,
        )
        return {
            "transport_plans": couplings,
            "barycenter": barycenter,
            "us_final": us,
            "vs_final": vs,
            "iterations": i_final,
            "error": final_err,
        }

@partial(jax.jit, static_argnums=(4, 5))
def _solve_pdlp_barycenter(
    cost: jnp.ndarray,
    marginals: Sequence[ArrayLike],
    weights: ArrayLike,
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
        termination_evaluation_frequency=64,
    )
    M = len(marginals)

    problem = create_barycenter_problem(cost, marginals, weights)
    result, _ = solver.optimize(problem, dim=problem.n)

    couplings = result.primal_solution.P
    barycenter = result.primal_solution.a
    us = result.dual_solution[:M]
    vs = result.dual_solution[M:]
    iters = result.iteration_count
    error = result.primal_residual_norm

    return couplings, barycenter, us, vs, iters, error