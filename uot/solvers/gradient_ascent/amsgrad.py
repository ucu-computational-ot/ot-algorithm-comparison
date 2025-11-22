from collections.abc import Sequence
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
import optax

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike
from uot.solvers.gradient_ascent._make_schedule import _make_schedule


class AMSGradSolver(BaseSolver):
    """
    2-marginal dual gradient ascent for entropic OT using AMSGrad (+ schedule).
    Cost matrix C has shape (n, m) matching marginal weights a (n,), b (m,).
    Convergence based on L2 residual of row/col sums.
    """
    def __init__(
        self,
        learning_rate: float = 1e-3,
        schedule: str = "constant",
        schedule_kwargs: dict = {},
    ):
        super().__init__()
        self.init_lr = float(learning_rate)
        self.schedule_name = schedule
        self.schedule_kwargs = dict(schedule_kwargs)  # shallow copy
        # Keep optimizer at LR=1.0; scale updates by schedule each step.
        self.optimizer = optax.amsgrad(learning_rate=1.0)

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],   # [mu, nu]
        costs: Sequence[ArrayLike],            # [C]
        reg: float,
        maxiter: int,
        tol: float,
        *args,
        normalize_cost: bool = True,
        **kwargs,
    ) -> dict:
        # --- extract & normalize marginals (stable) ---
        a = jnp.asarray(marginals[0].to_discrete()[1])
        b = jnp.asarray(marginals[1].to_discrete()[1])
        a = jnp.clip(a / jnp.sum(a), a_min=1e-10)
        b = jnp.clip(b / jnp.sum(b), a_min=1e-10)

        # --- cost & shapes ---
        C = jnp.asarray(costs[0])
        if C.shape != (a.shape[0], b.shape[0]):
            raise ValueError(f"Cost shape {C.shape} incompatible with marginals {(a.shape[0], b.shape[0])}.")

        # --- optional cost normalization ---
        if normalize_cost:
            scaling = jnp.max(jnp.abs(C))
            scaling = jnp.where(scaling > 0, scaling, 1.0)
            C_norm = C / scaling
            eps = reg / scaling
        else:
            C_norm = C
            eps = reg

        # --- per-call schedule tied to this run ---
        sched_kwargs = dict(self.schedule_kwargs)
        sched_kwargs.setdefault("decay_steps", int(maxiter))
        schedule_fn = _make_schedule(self.schedule_name, self.init_lr, **sched_kwargs)

        # --- init potentials & optimizer state outside JIT ---
        u0 = jnp.zeros_like(a)
        v0 = jnp.zeros_like(b)
        opt_state0 = self.optimizer.init((u0, v0))

        # --- run JIT core ---
        plan, cost, u_fin, v_fin, iters, err = _ga_amsgrad_2d(
            a=a, b=b, C=C_norm, eps=eps,
            maxiter=maxiter, tol=tol,
            opt_state=opt_state0,
            optimizer=self.optimizer,
            schedule=schedule_fn,
        )

        if normalize_cost:
            cost = cost * scaling

        return {
            "transport_plan": plan,
            "cost": cost,
            "u_final": u_fin,
            "v_final": v_fin,
            "iterations": iters,
            "error": err,
        }


# -------------------- JIT core (2-marginal) --------------------

def _logsumexp(x: jnp.ndarray, axis=None, keepdims=False) -> jnp.ndarray:
    m = jnp.max(x, axis=axis, keepdims=True)
    lse = m + jnp.log(jnp.sum(jnp.exp(x - m), axis=axis, keepdims=True) + 1e-12)
    if not keepdims and axis is not None:
        lse = jnp.squeeze(lse, axis=axis)
    return lse

@partial(jax.jit, static_argnames=("maxiter", "tol", "optimizer", "schedule"))
def _ga_amsgrad_2d(
    *,
    a: jnp.ndarray,                  # (n,)
    b: jnp.ndarray,                  # (m,)
    C: jnp.ndarray,                  # (n, m)
    eps: float,                      # entropic reg
    maxiter: int,
    tol: float,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    schedule: optax.Schedule,
):
    # init potentials
    u0 = jnp.zeros_like(a)
    v0 = jnp.zeros_like(b)

    def cond_fn(state):
        i, u, v, opt_state, err = state
        return (err > tol) & (i < maxiter)

    def body_fn(state):
        i, u, v, opt_state, _ = state

        # log_K = (u ⊕ v - C) / eps
        log_K = (u[:, None] + v[None, :] - C) / eps

        # gradients for ascent: grad_u = a - row_sums(P), grad_v = b - col_sums(P)
        # where P = exp(log_K)
        # use LSE then exp to avoid overflow on P directly
        row_lse = _logsumexp(log_K, axis=1)     # (n,)
        col_lse = _logsumexp(log_K, axis=0)     # (m,)
        row_sums = jnp.exp(row_lse)
        col_sums = jnp.exp(col_lse)

        grad_u = a - row_sums
        grad_v = b - col_sums

        # optax is descent → negate grads for ascent
        updates, opt_state = optimizer.update((-grad_u, -grad_v), opt_state, (u, v))

        # apply schedule per step
        step_lr = schedule(i)  # scalar
        updates = jax.tree.map(lambda g: g * step_lr, updates)

        # update potentials
        u, v = optax.apply_updates((u, v), updates)

        # gauge centering (mean-zero sum across both potentials)
        # mu, mv = jnp.mean(u), jnp.mean(v)
        # mbar = 0.5 * (mu + mv)
        # u = u - (mu - mbar)
        # v = v - (mv - mbar)

        # recompute residual with updated (u, v)
        log_K2 = (u[:, None] + v[None, :] - C) / eps
        row_err = jnp.linalg.norm(jnp.exp(_logsumexp(log_K2, axis=1)) - a, ord=2)
        col_err = jnp.linalg.norm(jnp.exp(_logsumexp(log_K2, axis=0)) - b, ord=2)
        err = jnp.maximum(row_err, col_err)

        return (i + 1, u, v, opt_state, err)

    i_fin, u_fin, v_fin, opt_state_fin, final_err = lax.while_loop(
        cond_fn, body_fn, (0, u0, v0, opt_state, jnp.inf)
    )

    # final plan & cost
    log_K_final = (u_fin[:, None] + v_fin[None, :] - C) / eps
    P_final = jnp.exp(log_K_final)
    cost = jnp.sum(P_final * C)

    return P_final, cost, u_fin, v_fin, i_fin, final_err




# from collections.abc import Sequence
# from functools import partial
# from typing import Tuple, List

# import jax
# import jax.numpy as jnp
# from jax import lax
# import optax

# from uot.data.measure import DiscreteMeasure
# from uot.solvers.base_solver import BaseSolver
# from uot.utils.types import ArrayLike
# from uot.solvers.gradient_ascent._make_schedule import _make_schedule


# class AMSGradSolver(BaseSolver):
#     """
#     Multi-marginal (N<=3) dual gradient ascent for entropic OT using AMSGrad.
#     - Works for 1D, 2D, 3D in the sense of number of marginals (N=1,2,3).
#     - Cost tensor must have shape (n1, ... , nN) matching the N marginals.
#     - Convergence based on L2 marginal residual only.
#     """
#     def __init__(
#             self,
#             learning_rate: float = 1e-3,
#             schedule: str = "constant",
#             schedule_kwargs: dict = {},
#             ):
#         super().__init__()
#         # self.learning_rate = learning_rate  # default; can be overridden per-call
#         self.init_lr = float(learning_rate)
#         self.schedule_name = schedule
#         self.schedule_kwargs = dict(schedule_kwargs)  # shallow copy

#         # AMSGrad with unit LR; we scale updates by the schedule each step
#         self.optimizer = optax.amsgrad(learning_rate=1.0)
#         self.schedule = _make_schedule(schedule, self.init_lr, **self.schedule_kwargs)


#     def solve(
#         self,
#         marginals: Sequence[DiscreteMeasure],
#         costs: Sequence[ArrayLike],
#         reg: float,
#         maxiter: int,
#         tol: float,
#         # learning_rate: float | None = None,
#         normalize_cost: bool = True,
#         *args,
#         **kwargs,
#     ) -> dict:
#         # Extract weights only (locations are irrelevant for the dual)
#         a_list: List[jnp.ndarray] = [jnp.asarray(m.to_discrete()[1]) for m in marginals]
#         N = len(a_list)
#         if N < 1 or N > 3:
#             raise ValueError(f"This implementation supports 1 to 3 marginals, got N={N}.")

#         C = jnp.asarray(costs[0])
#         expected_shape = tuple(arr.shape[0] for arr in a_list)
#         if C.shape != expected_shape:
#             raise ValueError(f"Cost shape {C.shape} incompatible with marginals {expected_shape}.")
        
#         us0 = tuple(jnp.zeros_like(a_list[i]) for i in range(N))
#         opt_state0 = self.optimizer.init(us0)

#         # lr = self.learning_rate if learning_rate is None else learning_rate

#         plan, cost, potentials, iters, err = _mm_gradient_amsgrad(
#             a_list=a_list,
#             C=C,
#             eps=reg,
#             maxiter=maxiter,
#             tol=tol,
#             # learning_rate=lr
#             opt_state=opt_state0,
#             optimizer=self.optimizer,
#             schedule=self.schedule,
#         )

#         # Return per-marginal potentials u_i
#         out = {
#             "transport_plan": plan,  # shape (n1,...,nN)
#             "cost": cost,
#             # "potentials": potentials,  # list/tuple of u_i (each shape (ni,))
#             "iterations": iters,
#             "error": err,             # L2 residual (max over marginals)
#         }
#         if N == 2:      # 2-marginal case is the most common
#             out.update({
#                 "u_final": potentials[0],
#                 "v_final": potentials[1],
#             })
#         else:
#             out.update({
#                 "potentials": potentials,
#             })
#         return out


# # ---------- helpers (log-sum-exp, gauge, broadcasting) ----------

# def _axes_except(i: int, N: int) -> Tuple[int, ...]:
#     return tuple(ax for ax in range(N) if ax != i)

# def _reshape_for_axis(u: jnp.ndarray, axis: int, N: int) -> jnp.ndarray:
#     # reshape 1D potential u (len ni) into shape (1,..,1, ni, 1,..,1) with ni at "axis"
#     shape = [1] * N
#     shape[axis] = u.shape[0]
#     return u.reshape(shape)

# def _logsumexp_over_axes(x: jnp.ndarray, axes: Tuple[int, ...]) -> jnp.ndarray:
#     # Stable log-sum-exp over multiple axes
#     if len(axes) == 0:
#         return x  # nothing to reduce
#     m = jnp.max(x, axis=axes, keepdims=True)
#     lse = m + jnp.log(jnp.sum(jnp.exp(x - m), axis=axes, keepdims=True))
#     # squeeze exactly those axes
#     for ax in sorted(axes, reverse=True):
#         lse = jnp.squeeze(lse, axis=ax)
#     return lse

# def _center_gauge_multi(us: Tuple[jnp.ndarray, ...]) -> Tuple[jnp.ndarray, ...]:
#     """
#     Center the gauge across N marginals: shift each u_i by (mean(u_i) - average_of_means),
#     so that the total sum of shifts is zero (keeps sum_i u_i unchanged).
#     """
#     means = [jnp.mean(u) for u in us]
#     mbar = sum(means) / len(us)
#     shifts = [m - mbar for m in means]
#     return tuple(u - s for u, s in zip(us, shifts))


# # ---------- residual & objective (log-domain, multi-marginal) ----------

# def _mm_marginal_sums_from_potentials(
#     us: Tuple[jnp.ndarray, ...], C: jnp.ndarray, eps: float
# ) -> Tuple[Tuple[jnp.ndarray, ...], jnp.ndarray]:
#     """
#     Given potentials (u1,...,uN), compute for each i the marginal sum along all other axes:
#        m_i[x_i] = sum_{x_{-i}} exp((sum_k u_k[x_k] - C[x_1,...,x_N]) / eps)
#     Returns (tuple of m_i arrays), and optionally reuse the log_K if needed.
#     """
#     N = len(us)
#     # log_K shape = C.shape
#     log_K = -C / eps
#     for i, u in enumerate(us):
#         log_K = log_K + _reshape_for_axis(u / eps, i, N)

#     marg_sums = []
#     for i in range(N):
#         axes = _axes_except(i, N)
#         lse_i = _logsumexp_over_axes(log_K, axes)  # shape (n_i,)
#         m_i = jnp.exp(lse_i)                        # row/col/... sum for marginal i
#         marg_sums.append(m_i)
#     return tuple(marg_sums), log_K

# def _mm_residual_l2(
#     marg_sums: Tuple[jnp.ndarray, ...], a_list: Tuple[jnp.ndarray, ...]
# ) -> jnp.ndarray:
#     # L2 residual per marginal; return the max over marginals
#     errs = [jnp.linalg.norm(ms - a, ord=2) for ms, a in zip(marg_sums, a_list)]
#     return jnp.max(jnp.stack(errs))


# # ---------- main jitted loop (AMSGrad) ----------

# @partial(jax.jit, static_argnames=("maxiter", "tol", "optimizer", "schedule"))
# def _mm_gradient_amsgrad(
#     a_list: Tuple[jnp.ndarray, ...],
#     C: jnp.ndarray,
#     eps: float,
#     *,
#     maxiter: int,
#     tol: float,
#     opt_state: optax.OptState,
#     optimizer: optax.GradientTransformation,
#     schedule: optax.Schedule,
# ):
#     """
#     Multi-marginal (N<=3) dual gradient ascent with AMSGrad.
#     Returns:
#       plan (exp log_K), cost, tuple(potentials), iterations, final_residual
#     """
#     # Convert to tuple for PyTree-friendliness
#     a_tuple: Tuple[jnp.ndarray, ...] = tuple(a_list)
#     N = len(a_tuple)

#     # initialize potentials u_i (each shape (n_i,))
#     us0 = tuple(jnp.zeros_like(a_tuple[i]) for i in range(N))

#     i0, res0 = 0, jnp.inf

#     # optimizer inside jit (lr is static so recompiles if changed)
#     # optimizer = optax.amsgrad(learning_rate=learning_rate)
#     # opt_state0 = optimizer.init(us0)

#     def cond_fn(state):
#         i, us, opt_state, res = state
#         return (res > tol) & (i < maxiter)

#     def body_fn(state):
#         i, us, opt_state, _ = state

#         # Compute marginal sums from current potentials (stable)
#         marg_sums, log_K = _mm_marginal_sums_from_potentials(us, C, eps)

#         # ASCENT gradients: grad_u_i = a_i - marginal_sum_i
#         grads = tuple(a - ms for a, ms in zip(a_tuple, marg_sums))

#         # Optax does DESCENT; negate grads for ASCENT
#         updates, opt_state = optimizer.update(tuple(-g for g in grads), opt_state, us)
#         # Apply schedule: scale updates by step-specific lr
#         step_lr = schedule(i)  # scalar array
#         updates = jax.tree.map(lambda u: u * step_lr, updates)
#         us = optax.apply_updates(us, updates)

#         # Gauge centering with zero-sum shifts across marginals
#         us = _center_gauge_multi(us)

#         # Recompute residual cheaply from the same log_K? We need updated us; recompute:
#         marg_sums_new, _ = _mm_marginal_sums_from_potentials(us, C, eps)
#         res = _mm_residual_l2(marg_sums_new, a_tuple)

#         return (i + 1, us, opt_state, res)

#     i_final, us_final, opt_state_final, final_res = lax.while_loop(
#         cond_fn, body_fn, (i0, us0, opt_state, res0)
#     )

#     # Build plan and cost at the end (OK to exponentiate once)
#     _, log_K_final = _mm_marginal_sums_from_potentials(us_final, C, eps)
#     plan = jnp.exp(log_K_final)  # shape matches C
#     cost = jnp.sum(plan * C)

#     return plan, cost, us_final, i_final, final_res
