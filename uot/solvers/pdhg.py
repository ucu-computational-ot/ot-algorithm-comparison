import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from functools import partial
from matplotlib import pyplot as plt

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike

from typing import Sequence


class OTState(NamedTuple):
    """
    NamedTuple to hold the state during PDHG iteration for OT.

    Attributes
    ----------
    coupling_k : jax.Array
        Current coupling (transport plan), shape (n, m).
    u_k : jax.Array
        Dual vector for the "mu" (row) constraints, shape (n,).
    v_k : jax.Array
        Dual vector for the "nu" (column) constraints, shape (m,).
    computed_marginals : tuple
        Cached row/column sums from the coupling, i.e. (row_sums, col_sums).
    k : int
        Current iteration count.
    done : bool
        Whether the stopping criterion has been met.
    """
    coupling_k: jax.Array
    u_k: jax.Array
    v_k: jax.Array
    computed_marginals: Tuple[jax.Array, jax.Array]
    k: int

# ==============================================================================
#  PDHG step functions
# ==========xw====================================================================

@jax.jit
def positive_marginals_vmap(u, v, C, eps):
    def row_fn(ui, Ci_row):
        # Ci_row is shape (m,)
        return jnp.sum(jnp.clip(ui + v - Ci_row, a_min=0.0))
    # vmapped row sums: shape (n,)
    row_raw = jax.vmap(row_fn)(u, C)
    # column sums: sum the transpose
    col_raw = jax.vmap(row_fn)(v, C.T)
    return row_raw / eps, col_raw / eps

@jax.jit
def pdhg_quadratic_ot_step(
        coupling_k: jax.Array,
        u: jax.Array,
        v: jax.Array,
        computed_marginals_prev: Tuple[jax.Array, jax.Array],
        C: jax.Array,
        mu: jax.Array,
        nu: jax.Array,
        tau: float,
        sigma: float,
        eps: float,
) -> Tuple[jax.Array, jax.Array, jax.Array, Tuple[jax.Array, jax.Array]]:
    """
    One iteration of PDHG for the Quadratic-regularized OT problem.

    We perform:
      1) coupling^(k+1) = prox_quadratic_coupling(coupling^k, u^k, v^k, C, tau, eps)
      2) u^(k+1), v^(k+1) = dual updates using row/column marginals.

    Parameters
    ----------
    coupling_k : jax.Array
        Current coupling (n, m).
    u : jax.Array
        Current dual vector for row constraints (n,).
    v : jax.Array
        Current dual vector for column constraints (m,).
    computed_marginals_prev : Tuple[jax.Array, jax.Array]
        (row_sum(coupling_k), col_sum(coupling_k)).
    C : jax.Array
        Cost matrix (n, m).
    mu : jax.Array
        Vector of supply constraints (n,).
    nu : jax.Array
        Vector of demand constraints (m,).
    tau : float
        Primal step size.
    sigma : float
        Dual step size.
    eps : float
        Quadratic regularization weight.

    Returns
    -------
    coupling_next : jax.Array
        Updated coupling (n, m).
    u_next : jax.Array
        Updated dual vector for row constraints (n,).
    v_next : jax.Array
        Updated dual vector for column constraints (m,).
    computed_marginals_next : Tuple[jax.Array, jax.Array]
        (row_sum, col_sum) of coupling_next.
    """
    # 1) Primal update
    grad = C - (u[:, None] + v[None, :])
    coupling_unconstrained = (coupling_k - tau * grad) / (1.0 + tau * eps)
    coupling_next = jnp.clip(coupling_unconstrained, 0.0, None)
    # X = u[:, None] + v[None, :] - C
    # coupling_next = jnp.clip(u[:, None] + v[None, :] - C, 0.0)

    # 2) Dual update — use the *new* marginals and the *old* marginals
    row_prev, col_prev = computed_marginals_prev
    # coupl = jnp.clip(u[:, None] + v[None, :] - C, 0.0)
    row_next = jnp.sum(coupling_next, axis=1)
    col_next = jnp.sum(coupling_next, axis=0)

    # row_next, col_next = positive_marginals_vmap(u, v, C, eps)

    # exactly:   2*row_next - row_prev  (no extra 2!)
    u_next = u + sigma * (mu - (2.0 * row_next - row_prev))
    v_next = v + sigma * (nu - (2.0 * col_next - col_prev))

    # coupling_next = jnp.clip(u[:, None] + v[None, :] - C, 0.0) / eps

    return coupling_next, u_next, v_next, (row_next, col_next), grad


# ==============================================================================
#  PDHG "main" functions for Quadratic vs. LP OT
# ==============================================================================
partial(jax.jit, static_argnums=(7, 8))
def pdhg_quadratic_ot(
        C: jax.Array,
        mu: jax.Array,
        nu: jax.Array,
        eps: float = 1e-2,
        tau: float = 0.9,
        sigma: float = 0.9,
        tol: float = 1e-8,
        max_outer_iter: int = 1000,
        # ref_coupling: jax.Array = None,
):
    """
    Run PDHG to solve the Quadratic-regularized OT problem:

      min_{coupling >= 0}  <coupling, C> + (eps/2) ||coupling||^2
      subject to  coupling @ 1 = mu,  coupling^T @ 1 = nu.

    This code uses an iterative primal-dual hybrid gradient method with a
    "preconditioned" update. The loop is unrolled via `jax.lax.scan`.

    Parameters
    ----------
    C : jax.Array
        Cost matrix of shape (n, m).
    mu : jax.Array
        Supply vector (size n).
    nu : jax.Array
        Demand vector (size m).
    eps : float, optional
        Quadratic regularization weight, by default 1e-2.
    tau : float, optional
        Primal step size, by default 0.9.
    sigma : float, optional
        Dual step size, by default 0.9.
    tol : float, optional
        Stopping tolerance on the maximum row/col constraints mismatch, by default 1e-8.
    max_outer_iter : int, optional
        Maximum number of PDHG iterations, by default 1000.
    initial_point : optional
        A tuple (coupling_init, dual_init) to initialize primal/dual. If None, uses a default guess.
        `dual_init` should be concatenated (u, v) or similar.
    save_iters : bool, optional
        If True, returns the coupling at each iteration from the scan, by default True.

    Returns
    -------
    coupling_final : jax.Array
        The final transport plan (n, m).
    u_final : jax.Array
        The final dual vector for row constraints (n,).
    v_final : jax.Array
        The final dual vector for column constraints (m,).
    iters : jax.Array or None
        If `save_iters=True`, returns a stack of coupling iterates. Otherwise None.

    Notes
    -----
    - GPU-ready: if you have JAX installed with CUDA support and your default device
      is GPU, it will automatically run there.
    - For extremely large problems, consider multi-GPU or HPC approaches.
    """
    n, m = C.shape

    # 1) Initialize primal/dual
    coupling_k = jnp.ones((n, m)) * (1.0 / (n * m))
    u = jnp.ones(n)
    v = jnp.ones(m)

    computed_marginals = (
        jnp.sum(coupling_k, axis=1),  # row sums
        jnp.sum(coupling_k, axis=0)  # column sums
    )

    mu = jnp.asarray(mu)
    nu = jnp.asarray(nu)

    @jax.jit
    def cond_fn(state):
        row_sum, col_sum = state.computed_marginals

        marginals_error = jnp.maximum(
            jnp.linalg.norm(row_sum - mu),
            jnp.linalg.norm(col_sum - nu)
        )
        return jnp.logical_or(
            jnp.logical_and(marginals_error >= tol, state.k < max_outer_iter),
            state.k < 5
        )

    # 2) Use lax.scan to unroll up to max_outer_iter
    init_state = OTState(coupling_k, u, v, computed_marginals, 0)

    def one_step(state):
        coupling_k, u_k, v_k, cmarg_prev, k = state
        coupling_next, u_next, v_next, cmarg_next, grad = pdhg_quadratic_ot_step(
            coupling_k, u_k, v_k, cmarg_prev, C, mu, nu, tau, sigma, eps
        )
        row_sum, col_sum = state.computed_marginals
        marginals_error = jnp.maximum(
            jnp.linalg.norm(row_sum - mu),
            jnp.linalg.norm(col_sum - nu)
        )
        # primal = jnp.sum(C * coupling_next) + 0.5 * eps * jnp.sum(coupling_next**2)
        # dual = u_next @ mu + v_next @ nu - (0.5 / eps) * jnp.sum(jnp.clip(-grad, 0.0)**2)
        stats = {
            "primal_feas": marginals_error,
            # "obj_diff": jnp.abs(jnp.sum(C * coupling_next) - jnp.sum(C * ref_coupling)),
            # "l2_diff": jnp.linalg.norm(ref_coupling - coupling_next),
            "dual_feas": jnp.linalg.norm(coupling_k - jnp.clip((-grad)/eps, 0.0))

        }
        return OTState(coupling_next, u_next, v_next, cmarg_next, k + 1)

    cpl_fin, u_fin, v_fin, cmarg_fin, final_iter = jax.lax.while_loop(
        cond_fn,
        one_step,
        init_state
    )

    # (cpl_fin, u_fin, v_fin, cmarg_fin, final_iter), iters = jax.lax.scan(
    #     one_step,
    #     init_state,
    #     xs=None,
    #     length=max_outer_iter,
    # )

    return cpl_fin, u_fin, v_fin, final_iter
#
#
# class PDHGSolver(BaseSolver):
#     def __init__(self):
#         return super().__init__()
#
#     def solve(
#         self,
#         marginals: Sequence[DiscreteMeasure],
#         costs: Sequence[ArrayLike],
#         reg: float = 1e-3,
#         maxiter: int = 1000,
#         tol: float = 1e-6,
#     ) -> dict:
#         if len(marginals) != 2:
#             raise ValueError("PDLP solver accepts only two marginals.")
#         if len(costs) == 0:
#             raise ValueError("Cost tensors not defined.")
#
#         mu, nu = marginals
#
#         problem = create_ot_problem(
#             C=costs[0],
#             mu=mu.to_discrete()[1],
#             nu=nu.to_discrete()[1],
#         )
#
#         coupling, u, v, i_final, final_err = _solve_pdlp(
#             problem=problem,
#             epsilon=reg,
#             precision=tol,
#             max_iters=maxiter,
#         )
#         return {
#             "transport_plan": coupling,
#             "u_final": u,
#             "v_final": v,
#             "iterations": i_final,
#             "error": final_err,
#         }