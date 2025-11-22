import jax
from jax import numpy as jnp
from jaxopt import LBFGS

from collections.abc import Sequence
from uot.solvers.base_solver import BaseSolver
from uot.data.measure import DiscreteMeasure
from uot.utils.types import ArrayLike


@jax.jit
def lbfgs(
    marginals: jnp.ndarray,
    C: jnp.ndarray,
    epsilon: float,
    # core parameters
    tolerance: float,
    maxiter: int,
    history_size: int = 15,
    use_gamma: bool = True,
    # line-search parameters
    maxls: int = 15,            # max inner iterations in LS
    stepsize: float = 0.0       # if \leq 0 runs a line-search, otherwise takes a fixed step
):
    a, b = marginals[0], marginals[1]

    def loss(carry):
        u, v = carry
        K = jnp.exp(
            (u[:, None] + v[None, :] - C) / epsilon
        )
        loss = -jnp.dot(a, u) - jnp.dot(b, v) + epsilon * jnp.sum(K)
        grad = (
            -a + jnp.sum(K, axis=1),
            -b + jnp.sum(K, axis=0),
        )
        return loss, grad

    carry = (jnp.zeros_like(a), jnp.zeros_like(b))
    solver = LBFGS(
        fun=loss,
        maxiter=maxiter,
        tol=tolerance,
        history_size=50,
        value_and_grad=True,
        # implicit_diff=False,
        use_gamma=True,
    )

    solver_result = solver.run(carry)
    u_opt, v_opt = solver_result.params
    P = jnp.exp(
        (u_opt[:, None] + v_opt[None, :] - C) / epsilon
    )
    return (u_opt, v_opt, P, solver_result.state.iter_num,
            solver_result.state.error)


class LBFGSPureSolver(BaseSolver):
    def __init__(self):
        super().__init__()

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        reg: float = 1e-3,
        maxiter: int = 1000,
        tol: float = 1e-6,
        history_size: int = 15,
        use_gamma: bool = True,
        # line-search parameters
        maxls: int = 15,            # max inner iterations in LS
        stepsize: float = 0.0       # if \leq 0 runs a line-search, otherwise
                                    # takes a fixed step
    ) -> dict:
        mu, nu = (
            marginals[0].to_discrete()[1],
            marginals[1].to_discrete()[1],
        )

        u, v, transport_plan, iters, grad_norm = lbfgs(
            marginals=jnp.array([mu, nu]),
            C=costs[0],
            epsilon=reg,
            tolerance=tol,
            maxiter=maxiter,
            history_size=history_size,
            use_gamma=use_gamma,
            maxls=maxls,
            stepsize=stepsize,
        )
        marginal_error = jnp.maximum(
            jnp.linalg.norm(transport_plan.sum(axis=1) - mu),
            jnp.linalg.norm(transport_plan.sum(axis=0) - nu),
        )
        return {
            "transport_plan": transport_plan,
            "cost": (transport_plan * costs[0]).sum(),
            "u_final": u,
            "v_final": v,
            "iterations": iters,
            "error": marginal_error,
        }
