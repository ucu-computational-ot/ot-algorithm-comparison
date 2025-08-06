from collections.abc import Sequence
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike

from uot.utils.solver_helpers import tensor_marginals, coupling_tensor


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

        transport_plan = coupling_tensor(u, v, costs[0], reg)

        return {
            "transport_plan": transport_plan,
            "cost": (transport_plan * costs[0]).sum(),
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
    P = coupling_tensor(u, v, cost, epsilon)
    row_marginal, col_marginal = tensor_marginals(P)
    return jnp.max(
        jnp.array(
            [jnp.linalg.norm(a - row_marginal),
             jnp.linalg.norm(b - col_marginal)]
        )
    )


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
        
        C = costs[0]
        mu, nu = marginals[0].to_discrete()[1], marginals[1].to_discrete()[1]
        
        # ln_K = -C / reg
        # ln_mu = jnp.log(mu)
        # ln_nu = jnp.log(nu)
        # ln_u = jnp.ones_like(ln_mu)
        # ln_v = jnp.ones_like(ln_nu)

        # @jax.jit
        # def body(carry):
        #     ln_u, ln_v, i, err = carry
        #     ln_u = ln_mu - jax.scipy.special.logsumexp(ln_K + ln_v[:, None], axis=0)
        #     ln_v = ln_nu - jax.scipy.special.logsumexp(ln_K.T + ln_u[:, None], axis=0)
        #     return ln_u, ln_v, i + 1, compute_error(carry)
        
        # @jax.jit
        # def condition(carry):
        #     u_, v_, i_, err_ = carry
        #     return jnp.logical_and(err_ > tol, i_ < maxiter)
        
        # @jax.jit
        # def compute_error(carry):
        #     ln_u, ln_v, i, err = carry
        #     ln_P = ln_u[:, None] + ln_K + ln_v[None, :]
        #     P = jnp.exp(ln_P)
        #     row_sum = P.sum(axis=1)
        #     col_sum = P.sum(axis=0)
        #     err_row = jnp.linalg.norm(row_sum - mu)
        #     err_col = jnp.linalg.norm(col_sum - nu)
        #     err = jnp.maximum(err_row, err_col)
        #     return err

        # init_err = compute_error((ln_u, ln_v, 0, None))
        # ln_u_final, ln_v_final, i_final, final_err = jax.lax.while_loop(
        #     condition, body, (ln_u, ln_v, jnp.array(0), init_err)
        # )

        # transport_plan = jnp.exp(ln_u[:, None] + ln_K + ln_v[None, :])
    

        P, u_final, v_final, n_steps, err = sinkhorn_jax(
            mu=mu,
            nu=nu,
            C=C,
            maxiter=maxiter,
            tol=tol,
            epsilon=reg,
        )
        return {
            "transport_plan": P,
            "cost": (P * costs[0]).sum(),
            "u_final": u_final,
            "v_final": v_final,
            "iterations": n_steps,
            "error": err,
        }
    

@partial(jax.jit, static_argnums=(3,4))
def sinkhorn_jax(mu, nu, C, maxiter: int, tol: float, epsilon: float = 1e-3):
    """
    JAX‑jitted Sinkhorn with while_loop stopping on tolerance.
    maxiter and tol are static (compile‑time) arguments.
    """
    # Precompute log‑kernel and log‑marginals
    ln_K  = -C / epsilon              # (n, m)
    ln_mu = jnp.log(mu)               # (n,)
    ln_nu = jnp.log(nu)               # (m,)

    ln_u0 = jnp.zeros_like(ln_mu)
    ln_v0 = jnp.zeros_like(ln_nu)
    init_error = jnp.inf

    def cond_fn(carry):
        ln_u, ln_v, i, err = carry
        return (err > tol) & (i < maxiter)
    
    def compute_error(ln_u, ln_v):
        ln_P    = ln_u[:, None] + ln_K + ln_v[None, :]
        P       = jnp.exp(ln_P)
        row_err = jnp.linalg.norm(P.sum(axis=1) - mu)
        col_err = jnp.linalg.norm(P.sum(axis=0) - nu)
        return jnp.maximum(row_err, col_err)

    def body_fn(carry):
        ln_u, ln_v, i, err = carry
        ln_u = ln_mu - logsumexp(ln_K + ln_v[None, :], axis=1)
        ln_v = ln_nu - logsumexp(ln_K.T + ln_u[None, :], axis=1)
        
        # err_upd = jax.lax.cond(
        #     i % 10 == 0,
        #     lambda: _compute_error(ln_u, ln_v),
        #     lambda: err,
        # )
        ln_P    = ln_u[:, None] + ln_K + ln_v[None, :]
        P       = jnp.exp(ln_P)
        row_err = jnp.linalg.norm(P.sum(axis=1) - mu)
        col_err = jnp.linalg.norm(P.sum(axis=0) - nu)
        err     = jnp.maximum(row_err, col_err)

        return (ln_u, ln_v, i + 1, err)

    ln_u_final, ln_v_final, iters, final_err = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (ln_u0, ln_v0, 0, init_error)
    )
    P_final = jnp.exp(ln_u_final[:, None] + ln_K + ln_v_final[None, :])
    return P_final, ln_u_final, ln_v_final, iters, final_err