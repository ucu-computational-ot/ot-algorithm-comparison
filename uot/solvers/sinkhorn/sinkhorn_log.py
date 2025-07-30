from collections.abc import Sequence
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike


class SinkhornTwoMarginalLogJaxSolver(BaseSolver):
    def __init__(self):
        super().__init__()

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
        # with the normalized cost sinkhorn performs MUCH faster
        C = costs[0] / costs[0].sum()
        mu, nu = marginals[0].to_discrete()[1], marginals[1].to_discrete()[1]

        P, cost, phi, psi, n_steps, err = sinkhorn_jax(
            mu=mu,
            nu=nu,
            C=C,
            maxiter=maxiter,
            tol=tol,
            epsilon=reg,
        )
        return {
            "transport_plan": P,
            "cost": jnp.sum(P * costs[0]),
            # "u_final": u,
            # "v_final": v,
            "iterations": n_steps,
            "error": err,
        }


@partial(jax.jit, static_argnums=(3, 4))
def sinkhorn_jax(mu, nu, C, maxiter: int, tol: float, epsilon: float = 1e-3):
    """
    JAX‑jitted Sinkhorn with while_loop stopping on tolerance.
    maxiter and tol are static (compile‑time) arguments.
    """
    # Precompute log‑kernel and log‑marginals
    ln_K = -C / epsilon              # (n, m)
    ln_mu = jnp.log(mu)               # (n,)
    ln_nu = jnp.log(nu)               # (m,)

    ln_u0 = jnp.zeros_like(ln_mu)
    ln_v0 = jnp.zeros_like(ln_nu)
    init_error = jnp.inf

    def cond_fn(carry):
        ln_u, ln_v, i, err = carry
        return (err > tol) & (i < maxiter)

    def compute_error(ln_u, ln_v):
        P = jnp.exp(ln_u[:, None] + ln_K + ln_v[None, :])
        return jnp.maximum(
            jnp.linalg.norm(P.sum(axis=1) - mu),
            jnp.linalg.norm(P.sum(axis=0) - nu),
        )

    def body_fn(carry):
        ln_u, ln_v, i, err = carry
        ln_u = ln_mu - logsumexp(ln_K + ln_v[None, :], axis=1)
        ln_v = ln_nu - logsumexp(ln_K.T + ln_u[None, :], axis=1)

        err = jax.lax.cond(
            i % 10 == 0,
            lambda: compute_error(ln_u, ln_v),
            lambda: err,
            operand=None,
        )

        return (ln_u, ln_v, i + 1, err)

    ln_u_final, ln_v_final, iters, final_err = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (ln_u0, ln_v0, 0, init_error)
    )
    P_final = jnp.exp(ln_u_final[:, None] + ln_K + ln_v_final[None, :])
    cost = jnp.sum(P_final * C)
    return P_final, cost, ln_u_final, ln_v_final, iters, final_err
