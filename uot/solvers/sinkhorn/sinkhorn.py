from collections.abc import Sequence
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
        # with the normalized cost sinkhorn performs MUCH faster
        # C = costs[0] / costs[0].max()
        C = costs[0]
        # u, v, i_final, final_err = _sinkhorn(
        #     a=mu.to_discrete()[1],
        #     b=nu.to_discrete()[1],
        #     cost=C,
        #     epsilon=reg,
        #     precision=tol,
        #     max_iters=maxiter,
        # )

        # transport_plan = coupling_tensor(u, v, C, reg)

        # return {
        #     "transport_plan": transport_plan,
        #     "cost": (transport_plan * costs[0]).sum(),
        #     "u_final": u,
        #     "v_final": v,
        #     "iterations": i_final,
        #     "error": final_err,
        # }
        a = mu.to_discrete()[1]
        b = nu.to_discrete()[1]
        u, v, i_final, final_err = _sinkhorn_plain(
            a=a,
            b=b,
            cost=C,
            epsilon=reg,
            precision=tol,
            max_iters=maxiter,
        )

        transport_plan = coupling_from_scalings(u, v, C, reg)

        return {
            "transport_plan": transport_plan,
            "cost": (transport_plan * costs[0]).sum(),
            "u_final": u,  # scaling vector (not log-potential)
            "v_final": v,  # scaling vector (not log-potential)
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


@jax.jit
def coupling_from_scalings(
    u: jnp.ndarray,
    v: jnp.ndarray,
    cost: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    # Global shift improves numerical range without changing the OT solution:
    # K' = exp(-(C - c0)/eps) = exp(c0/eps) * exp(-C/eps); the factor is absorbed by scalings.
    # c0 = jnp.min(cost)
    # K = jnp.exp(-(cost - c0) / epsilon)
    K = jnp.exp(-cost / epsilon)
    return (u[:, None] * K) * v[None, :]


@jax.jit
def _sinkhorn_plain(
    a: jnp.ndarray,
    b: jnp.ndarray,
    cost: jnp.ndarray,
    epsilon: float = 1e-3,
    precision: float = 1e-6,
    max_iters: int = 10_000,
    check_every: int = 10,
):
    n = a.shape[0]
    m = b.shape[0]

    # Kernel
    c0 = jnp.min(cost)
    K = jnp.exp(-(cost - c0) / epsilon)

    # Initialize scaling vectors
    u = jnp.ones((n,), dtype=a.dtype)
    v = jnp.ones((m,), dtype=b.dtype)

    tiny = jnp.array(1e-32, dtype=a.dtype)

    def cond_fn(carry):
        u_, v_, i_, err_ = carry
        return jnp.logical_and(err_ > precision, i_ < max_iters)

    def body_fn(carry):
        u_, v_, i_, err_ = carry

        Kv = K @ v_
        u_new = a / jnp.maximum(Kv, tiny)

        KTu = K.T @ u_new
        v_new = b / jnp.maximum(KTu, tiny)

        err_new = jax.lax.cond(
            (i_ % check_every) == 0,
            lambda _: _compute_error_from_K(u_new, v_new, a, b, K),
            lambda _: err_,
            operand=None,
        )

        return (u_new, v_new, i_ + jnp.array(1, dtype=i_.dtype), err_new)

    init_err = _compute_error_from_K(u, v, a, b, K)

    u_final, v_final, i_final, final_err = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (u, v, jnp.array(0, dtype=jnp.int32), init_err),
    )

    return u_final, v_final, i_final, final_err


@jax.jit
def _compute_error_from_K(
    u: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    K: jnp.ndarray,
) -> float:
    P = (u[:, None] * K) * v[None, :]
    row_marginal, col_marginal = tensor_marginals(P)
    return jnp.max(
        jnp.array(
            [
                jnp.linalg.norm(a - row_marginal),
                jnp.linalg.norm(b - col_marginal),
            ]
        )
    )

