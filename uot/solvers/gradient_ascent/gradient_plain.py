from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike


class GradientAscentPlainLogSolver(BaseSolver):
    def __init__(self):
        super().__init__()

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        reg: float,            # ε: entropic regularization
        maxiter: int,          # maximum iterations
        tol: float,            # stopping tolerance on error
        learning_rate: float,  # α: gradient step size
    ) -> dict:
        # Unpack inputs
        a = marginals[0].to_discrete()[1]
        b = marginals[1].to_discrete()[1]
        C = costs[0]
        eps = reg
        alpha   = learning_rate

        plan, cost, u_final, v_final, iters, err = _gradient(
            a, b, C, reg, maxiter, tol, learning_rate
        )

        return {
            "transport_plan": plan,
            "cost": cost,
            "u_final": u_final,
            "v_final": v_final,
            "iterations": iters,
            "error": err,
        }


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _gradient(
    a: jnp.ndarray,
    b: jnp.ndarray,
    C: jnp.ndarray,
    eps: float,
    maxiter: int,
    tol: float,
    alpha: float,
):
    # initial conditions
    phi0 = jnp.zeros_like(a)
    psi0 = jnp.zeros_like(b)
    i0 = jnp.array(0)
    err0 = jnp.inf

    def cond_fn(carry):
        _, _, i, err = carry
        return (err > tol) & (i < maxiter)

    def body_fn(carry):
        phi, psi, i, _ = carry
        # log-domain update
        log_K = (phi[:, None] + psi[None, :] - C) / eps
        m1 = jnp.max(log_K, axis=1, keepdims=True)
        E1 = jnp.exp(log_K - m1) * jnp.exp(m1)
        grad_phi = a - E1.sum(axis=1)
        m2 = jnp.max(log_K, axis=0, keepdims=True)
        E2 = jnp.exp(log_K - m2) * jnp.exp(m2)
        grad_psi = b - E2.sum(axis=0)
        phi_new = phi + alpha * grad_phi
        psi_new = psi + alpha * grad_psi
        # error
        log_P = (phi_new[:, None] + psi_new[None, :] - C) / eps
        P = jnp.exp(log_P)
        err_row = jnp.linalg.norm(P.sum(axis=1) - a)
        err_col = jnp.linalg.norm(P.sum(axis=0) - b)
        err_new = jnp.maximum(err_row, err_col)
        return (phi_new, psi_new, i + 1, err_new)

    phi_final, psi_final, iters, final_err = lax.while_loop(
        cond_fn, body_fn, (phi0, psi0, i0, err0)
    )

    # build plan and cost
    plan = jnp.exp((phi_final[:, None] + psi_final[None, :] - C) / eps)
    cost = jnp.sum(plan * C)
    return plan, cost, phi_final, psi_final, iters, final_err