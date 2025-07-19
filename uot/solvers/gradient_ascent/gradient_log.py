from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import optax

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike
from uot.utils.solver_helpers import coupling_tensor


class LogDomainGradientAscentSolver(BaseSolver):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.optimizer = optax.adam(learning_rate)
        self.learning_rate = learning_rate

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        reg: float,
        maxiter: int,
        tol: float,
        learning_rate: float = 1e-3,
    ) -> dict:
        a = marginals[0].to_discrete()[1]
        b = marginals[1].to_discrete()[1]
        C = costs[0]
        # initialize optimizer state
        opt_state = self.optimizer.init((jnp.zeros_like(a), jnp.zeros_like(b)))
        # call jitted core
        plan, cost, phi, psi, iters, err = _gradient(
            a, b, C, reg, maxiter, tol, self.learning_rate, opt_state, self.optimizer
        )
        return {
            "transport_plan": plan,
            "cost": cost,
            "u_final": phi,
            "v_final": psi,
            "iterations": iters,
            "error": err,
        }
    

@partial(jax.jit, static_argnums=(3, 4, 5, 6, 8))
def _gradient(
    a: jnp.ndarray,
    b: jnp.ndarray,
    C: jnp.ndarray,
    eps: float,
    maxiter: int,
    tol: float,
    alpha: float,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, float]:
    """
    JIT‑compiled core loop for log-domain gradient ascent on entropic OT dual.
    Returns plan, cost, phi_final, psi_final, iterations, final_error.
    """
    phi0 = jnp.zeros_like(a)
    psi0 = jnp.zeros_like(b)
    i0, err0 = 0, jnp.inf

    def cond_fn(state):
        i, phi, psi, opt_state, err = state
        return (err > tol) & (i < maxiter)

    def body_fn(state):
        i, phi, psi, opt_state, _ = state
        # stabilized log-K
        log_K = (phi[:, None] + psi[None, :] - C) / eps
        m1 = jnp.max(log_K, axis=1, keepdims=True)
        E1 = jnp.exp(log_K - m1) * jnp.exp(m1)
        grad_phi = a - E1.sum(axis=1)
        m2 = jnp.max(log_K, axis=0, keepdims=True)
        E2 = jnp.exp(log_K - m2) * jnp.exp(m2)
        grad_psi = b - E2.sum(axis=0)
        # apply optimizer update
        grads = (-grad_phi, -grad_psi)
        updates, opt_state = optimizer.update(grads, opt_state, (phi, psi))
        phi, psi = optax.apply_updates((phi, psi), updates)
        # compute error
        log_P = (phi[:, None] + psi[None, :] - C) / eps
        P = jnp.exp(log_P)
        err_row = jnp.max(jnp.abs(P.sum(axis=1) - a))
        err_col = jnp.max(jnp.abs(P.sum(axis=0) - b))
        err = jnp.maximum(err_row, err_col)
        return (i+1, phi, psi, opt_state, err)

    # run while loop
    i_final, phi_final, psi_final, opt_state_final, final_err = lax.while_loop(
        cond_fn, body_fn, (i0, phi0, psi0, opt_state, err0)
    )
    plan = jnp.exp((phi_final[:, None] + psi_final[None, :] - C) / eps)
    cost = jnp.sum(plan * C)
    return plan, cost, phi_final, psi_final, i_final, final_err