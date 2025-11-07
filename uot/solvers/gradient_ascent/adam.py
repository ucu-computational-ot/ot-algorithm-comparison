from collections.abc import Sequence
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
import optax

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike

from uot.solvers.gradient_ascent._make_schedule import _make_schedule


# ----------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------
class AdamGradientAscentSolver(BaseSolver):
    """
    Dual ascent for entropic OT with Adam + *any* step-size schedule.
    """
    def __init__(
        self,
        learning_rate: float = 1e-3,
        schedule: str = "constant",
        schedule_kwargs: dict = {},
    ):
        super().__init__()
        # 1. base learning-rate (used as init_lr for the schedule)
        self.init_lr = learning_rate
        # 2. build the schedule object (optax.Schedule)
        self.schedule = _make_schedule(schedule, learning_rate, **schedule_kwargs)

        # 3. Adam with *learning-rate = 1.0* – the schedule will be applied
        #    inside the loop via `optax.apply_updates(updates * lr, ...)`
        self.optimizer = optax.adam(1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        reg: float,
        maxiter: int,
        tol: float,
        *args,
        normalize_cost: bool = True,
        **kwargs,
    ) -> dict:
        a = marginals[0].to_discrete()[1]
        b = marginals[1].to_discrete()[1]
        cost_original = costs[0]
        scaling = jnp.max(jnp.abs(cost_original)) if normalize_cost else 1.0
        C_norm = cost_original / scaling if normalize_cost else cost_original
        reg = float(reg / scaling if normalize_cost else reg)

        a = jnp.clip(a / jnp.sum(a), a_min=1e-10)
        b = jnp.clip(b / jnp.sum(b), a_min=1e-10)

        opt_state = self.optimizer.init((jnp.zeros_like(a), jnp.zeros_like(b)))
        
        plan, cost, phi, psi, iters, err = _gradient(
            a, b, C_norm, reg, maxiter, tol,
            opt_state, self.optimizer, self.schedule
        )
        cost = cost * scaling if normalize_cost else cost
        return {
            "transport_plan": plan,
            "cost": cost,
            "u_final": phi,
            "v_final": psi,
            "iterations": iters,
            "error": err,
        }

    # ------------------------------------------------------------------
    # LR-finder
    # ------------------------------------------------------------------
    @staticmethod
    def find_lr(
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        reg: float,
        *args,
        num_iters: int = 100,
        base_lr: float = 1e-7,
        max_lr: float = 10.0,
        **kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        a = marginals[0].to_discrete()[1]
        b = marginals[1].to_discrete()[1]
        C = costs[0]

        a = jnp.clip(a / jnp.sum(a), a_min=1e-10)
        b = jnp.clip(b / jnp.sum(b), a_min=1e-10)

        finder_opt = optax.adam(1.0)
        opt_state = finder_opt.init((jnp.zeros_like(a), jnp.zeros_like(b)))

        return _lr_finder(
            a, b, C, reg, num_iters, base_lr, max_lr,
            opt_state, finder_opt
        )


# ----------------------------------------------------------------------
# JIT-ted core (now receives the schedule)
# ----------------------------------------------------------------------
@partial(jax.jit, static_argnums=(3, 4, 5, 7, 8))
def _gradient(
    a: jnp.ndarray,
    b: jnp.ndarray,
    C: jnp.ndarray,
    eps: float,
    maxiter: int,
    tol: float,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    schedule: optax.Schedule,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, float]:
    phi0 = jnp.zeros_like(a)
    psi0 = jnp.zeros_like(b)
    i0, err0 = 0, jnp.inf

    def log_sum_exp(log_K, axis):
        max_val = jnp.max(log_K, axis=axis, keepdims=True)
        return max_val.squeeze(axis) + jnp.log(jnp.sum(jnp.exp(log_K - max_val), axis=axis) + 1e-10)

    def cond(state):
        i, _, _, _, err = state
        return (err > tol) & (i < maxiter)

    def body(state):
        i, phi, psi, opt_state, _ = state

        # ---- gradients -------------------------------------------------
        log_K = (phi[:, None] + psi[None, :] - C) / eps
        grad_phi = a - jnp.exp(log_sum_exp(log_K, axis=1))
        grad_psi = b - jnp.exp(log_sum_exp(log_K, axis=0))
        grads = (-grad_phi, -grad_psi)

        # ---- optimizer step (lr = 1.0) ---------------------------------
        updates, opt_state = optimizer.update(grads, opt_state, (phi, psi))

        # ---- apply *schedule* -----------------------------------------
        step_lr = schedule(i)                     # i is the current iteration
        phi, psi = optax.apply_updates((phi, psi), jax.tree.map(lambda u: u * step_lr, updates))

        # ---- error ----------------------------------------------------
        log_P = (phi[:, None] + psi[None, :] - C) / eps
        P = jnp.exp(log_P)
        err = jnp.maximum(jnp.max(jnp.abs(P.sum(1) - a)),
                          jnp.max(jnp.abs(P.sum(0) - b)))
        return i + 1, phi, psi, opt_state, err

    i_final, phi_f, psi_f, opt_state_f, final_err = lax.while_loop(
        cond, body, (i0, phi0, psi0, opt_state, err0)
    )

    plan = jnp.exp((phi_f[:, None] + psi_f[None, :] - C) / eps)
    cost = jnp.sum(plan * C)
    return plan, cost, phi_f, psi_f, i_final, final_err


# ----------------------------------------------------------------------
# LR-finder (unchanged)
# ----------------------------------------------------------------------
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 8))
def _lr_finder(
    a: jnp.ndarray,
    b: jnp.ndarray,
    C: jnp.ndarray,
    eps: float,
    num_iters: int,
    base_lr: float,
    max_lr: float,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    phi0 = jnp.zeros_like(a)
    psi0 = jnp.zeros_like(b)

    log_lrs = jnp.linspace(jnp.log(base_lr), jnp.log(max_lr), num_iters)
    lrs = jnp.exp(log_lrs)

    def compute_dual(phi, psi):
        log_K = (phi[:, None] + psi[None, :] - C) / eps
        max_val = jnp.max(log_K)
        exp_terms = jnp.exp(log_K - max_val)
        sum_exp = jnp.sum(exp_terms)
        logsumexp = max_val + jnp.log(sum_exp + 1e-10)
        dual = jnp.dot(phi, a) + jnp.dot(psi, b) - eps * jnp.exp(logsumexp)
        return dual

    def body(carry, i):
        phi, psi, opt_state = carry
        cur_lr = lrs[i]

        log_K = (phi[:, None] + psi[None, :] - C) / eps
        m1 = jnp.max(log_K, axis=1, keepdims=True)
        E1 = jnp.exp(log_K - m1) * jnp.exp(m1)
        grad_phi = a - E1.sum(axis=1)
        m2 = jnp.max(log_K, axis=0, keepdims=True)
        E2 = jnp.exp(log_K - m2) * jnp.exp(m2)
        grad_psi = b - E2.sum(axis=0)

        grads = (-grad_phi, -grad_psi)
        updates, opt_state = optimizer.update(grads, opt_state, (phi, psi))
        phi, psi = optax.apply_updates((phi, psi), jax.tree.map(lambda u: u * cur_lr, updates))

        loss = -compute_dual(phi, psi)
        return (phi, psi, opt_state), loss

    _, losses = lax.scan(body, (phi0, psi0, opt_state), jnp.arange(num_iters))
    return lrs, losses