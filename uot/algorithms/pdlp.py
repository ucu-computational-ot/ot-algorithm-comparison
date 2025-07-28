from jax import numpy as jnp
from .rapdhg import raPDHG, create_ot_problem
from .rapdhg.strategies.reg_strategy import RegStrategy
from .rapdhg.utils import RestartScheme
from .rapdhg.utils import OTProblem


def solve_pdlp(problem: OTProblem, epsilon=0.01, precision=1e-4, max_iters=20_000, final_eps=0.0):
    solver = raPDHG(
        verbose=False,
        jit=False,
        reg=float(epsilon),
        eps_abs=precision,
        eps_rel=precision,
        iteration_limit=max_iters,
        termination_evaluation_frequency=64,
    )
    problem = create_ot_problem(cost, a, b)
    result, _ = solver.optimize(problem, dim=problem.n)

    converged = result.termination_status == 2
    cost = jnp.sum(problem.cost_matrix * result.avg_primal_solution)

    return result.avg_primal_solution, cost, converged