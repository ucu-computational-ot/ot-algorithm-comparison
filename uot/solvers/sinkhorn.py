from typing import Sequence

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike


class SinkhornTwoMarginalSolver(BaseSolver):
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
        u, v, i_final, final_err = _sinkhorn(
            a=mu.to_discrete()[1],
            b=nu.to_discrete()[1],
            cost=costs[0],
            epsilon=reg,
            precision=tol,
            max_iters=maxiter,
        )
        return {
            "transport_plan": _coupling_tensor(u, v, costs[0], reg),
            "u_final": u,
            "v_final": v,
            "iterations": i_final,
            "error": final_err,
        }


@jax.jit
def _sinkhorn(
    a: jnp.ndarray,
    b: jnp.ndarray,
    cost: jnp.ndarray,
    epsilon: float = 1e-3,
    precision: float = 1e-4,
    max_iters: int = 10_000,
):
    n = a.shape[0]
    m = b.shape[0]

    # Initialize dual variables
    u = jnp.zeros(n)
    v = jnp.zeros(m)

    @jax.jit
    def cond_fn(carry):
        u_, v_, i_, err_ = carry
        return jnp.logical_and(err_ > precision, i_ < max_iters)

    @jax.jit
    def body_fn(carry):
        u_, v_, i_, e = carry

        u_upd = (
            u_
            + epsilon * jnp.log(a)
            - epsilon *
            logsumexp((u_[:, None] + v_[None, :] - cost) / epsilon, axis=1)
        )
        v_upd = (
            v_
            + epsilon * jnp.log(b)
            - epsilon
            * logsumexp((u_upd[:, None] + v_[None, :] - cost) / epsilon, axis=0)
        )

        err_upd = jax.lax.cond(
            i_ % 10 == 0,
            lambda: _compute_error(u_upd, v_upd, a, b, cost, epsilon),
            lambda: e,
        )

        return (u_upd, v_upd, i_ + 1, err_upd)

    init_err = _compute_error(u, v, a, b, cost, epsilon)
    u_final, v_final, i_final, final_err = jax.lax.while_loop(
        cond_fn, body_fn, (u, v, jnp.array(0), init_err)
    )

    return u_final, v_final, i_final, final_err


@jax.jit
def _compute_error(
    u: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    cost: jnp.ndarray,
    epsilon: float,
) -> float:
    P = _coupling_tensor(u, v, cost, epsilon)
    row_marginal, col_marginal = _tensor_marginals(P)
    return jnp.max(
        jnp.array(
            [jnp.linalg.norm(a - row_marginal),
             jnp.linalg.norm(b - col_marginal)]
        )
    )


@jax.jit
def _coupling_tensor(
    u: jnp.ndarray, v: jnp.ndarray, cost: jnp.ndarray, epsilon: float
) -> jnp.ndarray:
    return jnp.exp((u[:, None] + v[None, :] - cost) / epsilon)


@jax.jit
def _tensor_marginals(matrix: jnp.ndarray):
    row_marginal = jnp.sum(matrix, axis=1)  # sum over columns
    column_marginal = jnp.sum(matrix, axis=0)  # sum over rows
    return (row_marginal, column_marginal)
