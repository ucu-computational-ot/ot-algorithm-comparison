from collections.abc import Sequence
from functools import partial
from typing import Tuple, List

import jax
import jax.numpy as jnp
from jax import lax
import optax

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike


class GradientAscentMultiMarginalSGD(BaseSolver):
    """
    Entropic OT dual gradient ascent with SGD (+ momentum), N=1..3 marginals.

    - Cost tensor C must have shape (n1, ..., nN) for N in {1,2,3}.
    - Potentials are (u1, ..., uN), with ui.shape == (ni,).
    - Convergence: L2 marginal residual only (max over marginals).
    """
    def __init__(self,
                 learning_rate: float = 1e-3,
                 momentum: float = 0.9,
                 nesterov: bool = True):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        reg: float = 1e-3,
        maxiter: int = 50_000,
        tol: float = 1e-7,
        learning_rate: float | None = None,
        momentum: float | None = None,
        nesterov: bool | None = None,
    ) -> dict:
        # extract weights a_i
        a_list: List[jnp.ndarray] = [
            jnp.asarray(m.to_discrete(include_zeros=False)[1])
            for m in marginals
            ]
        N = len(a_list)
        if N < 1 or N > 3:
            raise ValueError(f"This implementation supports 1..3 marginals, got N={N}.")

        C = jnp.asarray(costs[0])
        expected_shape = tuple(ai.shape[0] for ai in a_list)
        if C.shape != expected_shape:
            raise ValueError(f"Cost shape {C.shape} incompatible with marginals {expected_shape}.")

        lr = float(self.learning_rate if learning_rate is None else learning_rate)
        mom = float(self.momentum if momentum is None else momentum)
        nes = bool(self.nesterov if nesterov is None else nesterov)

        plan, cost, potentials, iters, err = _mm_gradient_sgd(
            a_list=a_list, C=C, eps=float(reg),
            maxiter=int(maxiter), tol=float(tol),
            learning_rate=lr, momentum=mom, nesterov=nes,
        )

        return {
            "transport_plan": plan,      # shape (n1,...,nN)
            "cost": cost,                # scalar
            "potentials": potentials,    # tuple(u1,...,uN)
            "iterations": iters,         # int
            "residual_l2": err,          # scalar L2 residual (max over marginals)
        }


# -------------------- helpers --------------------

def _axes_except(i: int, N: int) -> Tuple[int, ...]:
    return tuple(ax for ax in range(N) if ax != i)

def _reshape_for_axis(u: jnp.ndarray, axis: int, N: int) -> jnp.ndarray:
    shape = [1] * N
    shape[axis] = u.shape[0]
    return u.reshape(shape)

def _logsumexp_over_axes(x: jnp.ndarray, axes: Tuple[int, ...]) -> jnp.ndarray:
    if len(axes) == 0:
        return x
    m = jnp.max(x, axis=axes, keepdims=True)
    lse = m + jnp.log(jnp.sum(jnp.exp(x - m), axis=axes, keepdims=True))
    # squeeze exactly the reduced axes
    for ax in sorted(axes, reverse=True):
        lse = jnp.squeeze(lse, axis=ax)
    return lse

def _center_gauge_multi(us: Tuple[jnp.ndarray, ...]) -> Tuple[jnp.ndarray, ...]:
    """
    Zero-sum gauge: shifts sum to zero across marginals to keep sum_i u_i unchanged.
    """
    means = [jnp.mean(u) for u in us]
    mbar = sum(means) / len(us)
    shifts = [m - mbar for m in means]
    return tuple(u - s for u, s in zip(us, shifts))

def _marginal_sums_from_potentials(
    us: Tuple[jnp.ndarray, ...],
    C: jnp.ndarray,
    eps: float,
) -> Tuple[Tuple[jnp.ndarray, ...], jnp.ndarray]:
    """
    Compute marginal sums m_i[x_i] = sum_{x_{-i}} exp((sum_k u_k[x_k] - C)/eps)
    using stable log-sum-exp; also return log_K for possible reuse.
    """
    N = len(us)
    log_K = -C / eps
    for i, u in enumerate(us):
        log_K = log_K + _reshape_for_axis(u / eps, i, N)

    marg_sums = []
    for i in range(N):
        axes = _axes_except(i, N)
        lse_i = _logsumexp_over_axes(log_K, axes)  # (n_i,)
        marg_sums.append(jnp.exp(lse_i))
    return tuple(marg_sums), log_K

def _residual_l2(marg_sums: Tuple[jnp.ndarray, ...], a_tuple: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    errs = [jnp.linalg.norm(ms - a, ord=2) for ms, a in zip(marg_sums, a_tuple)]
    return jnp.max(jnp.stack(errs))


# -------------------- main jitted loop (SGD+momentum) --------------------

@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def _mm_gradient_sgd(
    a_list: List[jnp.ndarray],
    C: jnp.ndarray,
    eps: float,
    maxiter: int,
    tol: float,
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
    nesterov: bool = True,
):
    """
    Multi-marginal (N<=3) dual gradient ascent with SGD(+momentum/Nesterov).
    Returns: (plan, cost, potentials_tuple, iterations, final_residual_L2)
    """
    a_tuple: Tuple[jnp.ndarray, ...] = tuple(a_list)
    N = len(a_tuple)

    # init potentials
    us0 = tuple(jnp.zeros_like(a_tuple[i]) for i in range(N))

    # build optimizer *inside* jit; args are static so safe
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    opt_state0 = optimizer.init(us0)

    def cond_fn(state):
        i, us, opt_state, res = state
        return (res > tol) & (i < maxiter)

    def body_fn(state):
        i, us, opt_state, _ = state

        # current marginal sums (stable)
        marg_sums, _ = _marginal_sums_from_potentials(us, C, eps)

        # ASCENT gradients: grad_u_i = a_i - marg_sum_i
        grads = tuple(a - ms for a, ms in zip(a_tuple, marg_sums))

        # Optax does DESCENT; negate grads for ASCENT
        updates, opt_state = optimizer.update(tuple(-g for g in grads), opt_state, us)
        us = optax.apply_updates(us, updates)

        # gauge centering (zero-sum across marginals)
        us = _center_gauge_multi(us)

        # residual for updated potentials
        marg_sums_new, _ = _marginal_sums_from_potentials(us, C, eps)
        res = _residual_l2(marg_sums_new, a_tuple)

        return (i + 1, us, opt_state, res)

    i_final, us_final, opt_state_final, final_res = lax.while_loop(
        cond_fn, body_fn, (0, us0, opt_state0, jnp.inf)
    )

    # final plan & cost
    _, log_K_final = _marginal_sums_from_potentials(us_final, C, eps)
    plan = jnp.exp(log_K_final)
    cost = jnp.sum(plan * C)

    return plan, cost, us_final, i_final, final_res


# from collections.abc import Sequence

# import jax
# import jax.numpy as jnp
# import optax

# from uot.utils.types import ArrayLike
# from uot.data.measure import DiscreteMeasure
# from uot.solvers.base_solver import BaseSolver

# from uot.utils.solver_helpers import coupling_tensor


# class GradientAscentTwoMarginalSolver(BaseSolver):
#     def __init__(self):
#         return super().__init__()

#     def solve(
#         self,
#         marginals: Sequence[DiscreteMeasure],
#         costs: Sequence[ArrayLike],
#         reg: float = 1e-3,
#         maxiter: int = 1000,
#         tol: float = 1e-6,
#         learning_rate: float = 1e-3,
#     ) -> dict:
#         if len(costs) == 0:
#             raise ValueError("Cost tensors not defined.")
#         if len(marginals) != 2:
#             raise ValueError("This gradient ascent solver accepts only two marginals.")
#         mu, nu = marginals[0].to_discrete()[1], marginals[1].to_discrete()[1]

#         marginals = jnp.array([mu, nu])

#         final_potentials, i_final, final_loss, final_err = gradient_ascent_opt_multimarginal(
#             marginals=marginals,
#             cost=costs[0],
#             eps=reg,
#             tol=tol,
#             max_iterations=maxiter,
#             learning_rate=learning_rate,
#         )

#         u, v = final_potentials[0][None, :].reshape(-1), final_potentials[1][:, None].reshape(-1)

#         transport_plan = coupling_tensor(u, v, costs[0], reg)

#         return {
#             "transport_plan": transport_plan,
#             "cost": (transport_plan * costs[0]).sum(),
#             "u_final": u,
#             "v_final": v,
#             "iterations": i_final,
#             "error": final_err,
#         }


# @jax.jit
# def gradient_ascent_opt_multimarginal(
#     marginals,
#     cost,
#     eps=1e-3,
#     learning_rate=1e-3,
#     max_iterations=100_000,
#     tol=1e-4,
# ) -> jax.Array:
#     N = marginals.shape[0]
#     n = marginals.shape[1]

#     shapes = [tuple(n if j == i else 1 for j in range(N)) for i in range(N)]
#     potentials = jnp.zeros_like(marginals)
#     optimizer = optax.sgd(learning_rate=learning_rate)
#     opt_state = optimizer.init(potentials)

#     @jax.jit
#     def objective(potentials: jax.Array):
#         """Computes the dual objective with logsumexp stabilization."""
#         potentials_reshaped = [potentials[i].reshape(shapes[i]) for i in range(N)]
#         potentials_sum = sum(potentials_reshaped)
#         log_sub_entropy = (potentials_sum - cost) / eps
#         max_log_sub_entropy = jnp.max(log_sub_entropy, axis=0, keepdims=True)
#         stable_sum = jnp.exp(max_log_sub_entropy) * jnp.sum(
#             jnp.exp(log_sub_entropy - max_log_sub_entropy), axis=0
#         )
#         dual = potentials * marginals
#         return jnp.sum(dual - eps * stable_sum)

#     objective_gradient = jax.value_and_grad(objective)

#     @jax.jit
#     def step(state: tuple[int, jax.Array, optax.OptState, float, float, bool]):
#         """Performs one gradient ascent step."""
#         i, potentials, opt_state, prev_loss, prev_err, _ = state
#         loss, grad = objective_gradient(potentials)
#         # minus gradient because we are performing gradient ascent
#         updates, opt_state = optimizer.update(-grad, opt_state, potentials)
#         potentials = optax.apply_updates(potentials, updates)
#         # L-infinity norm
#         max_change = jnp.max(jnp.abs(potentials - state[1]))
#         has_converged = max_change < tol
#         return i + 1, potentials, opt_state, loss, max_change, has_converged

#     def cond_fn(
#             state: tuple[int, jax.Array, optax.OptState, float, float, bool]
#     ):
#         i, _, _, _, _, has_converged = state
#         return jnp.logical_and(
#             i < max_iterations,
#             jnp.logical_not(has_converged),
#         )

#     final_state = jax.lax.while_loop(
#         cond_fn, step, (0, potentials, opt_state, jnp.inf, jnp.inf, False)
#     )

#     steps, final_potentials, _, final_loss, final_err, has_converged = final_state
#     return final_potentials, steps, final_loss, final_err