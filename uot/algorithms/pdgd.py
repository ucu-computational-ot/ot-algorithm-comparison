import jax.numpy as jnp
import jax
from jax import jit
import ot
import ot.plot
from jax.scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import NamedTuple
jax.config.update("jax_enable_x64", True)

class State(NamedTuple):
    P_k: jax.Array
    lam_k: jax.Array
    eta_k: jax.Array
    computed_marginals: tuple
    k: int
    done: bool

@jax.jit
def prox_quadratic_ot(
    Pk,    # current primal iterate (n x m)
    lam,   # dual vector for row constraints (size n)
    eta,   # dual vector for col constraints (size m)
    C,     # cost matrix (n x m)
    tau,   # primal stepsize
    eps,   # regularization weight
):
    """
    Primal update for the PDHG iteration with a quadratic penalty:
      min_{P>=0} <P, C> + (eps/2)||P||^2 + (1/(2tau))||P - [Pk - tau K^T tilde{y}^k]||^2
    """

    gradient = C + lam[:, None] + eta[None, :]

    denom = 1 + tau * eps
    P_unconstrained = (Pk - tau * gradient) / denom

    return jnp.clip(P_unconstrained, 0.0, jnp.inf)


@jax.jit
def pdhg_entropic_ot_step(
    Pk, lam, eta, computed_marginals_prev,
    C, alpha, beta,
    tau, sigma, eps,
):
    """
    One iteration of PDHG for entropic OT.
      1) P^{k+1} = prox_{tau*f}( P^k - tau * K^T tilde{y}^k )
      2) y^{k+1} = y^k + sigma * K(2P^{k+1} - P^k)
      3) store old y for next iteration
    Returns: (P^{k+1}, lam^{k+1}, eta^{k+1})
    """
    #=== 1) Primal update
    P_next = prox_quadratic_ot(Pk, lam, eta, C, tau, eps)

    #=== 2) Dual update

    # Previous update was less optimized
    # P_diff = 2.0*P_next - Pk

    computed_marginals_next = (
        jnp.sum(P_next, axis=1),
        jnp.sum(P_next, axis=0)
    )

    lam_next = lam + sigma*(2*computed_marginals_next[0] - computed_marginals_prev[0]  - alpha)
    eta_next = eta + sigma*(2*computed_marginals_next[1] - computed_marginals_prev[1] - beta)

    return P_next, lam_next, eta_next, computed_marginals_next



def pdhg_quadratic_ot(
    C, alpha, beta, eps=1e-2,
    tau=0.9, sigma=0.9, tol=1e-8,
    max_outer_iter=1000, initial_point=None, save_iters=True
):
    """
    Run PDHG to solve entropic OT:
       min_{P>=0} <P,C> + eps * sum_ij P_ij log(P_ij)
       s.t. P 1 = alpha, P^T 1 = beta.
    """
    n, m = C.shape
    # Initialize primal/dual
    Pk      = jnp.ones((n,m)) * (1.0/(n*m)) if initial_point is None else initial_point[0]
    lam     = jnp.zeros(n) if initial_point is None else initial_point[1][:n]
    eta     = jnp.zeros(m) if initial_point is None else initial_point[1][n:]

    computed_marginals = (
        jnp.sum(Pk, axis=1),
        jnp.sum(Pk, axis=0)
    )

    alpha = jnp.asarray(alpha)
    beta  = jnp.asarray(beta)

    @jax.jit
    def stopping_criteria(computed_marginals):
        marginal_1, marginal_2 = computed_marginals
        marginals_error = jnp.maximum(
            jnp.linalg.norm(marginal_1 - alpha),
            jnp.linalg.norm(marginal_2 - beta)
        )
        return tol > marginals_error

    def one_step(carry):
        (P_k, lam_k, eta_k, computed_marginals, k, done) = carry
        P_next, lam_next, eta_next, computed_marginals_next = pdhg_entropic_ot_step(
            P_k, lam_k, eta_k, computed_marginals, C, alpha, beta, tau, sigma, eps,
        )
        done = stopping_criteria(computed_marginals)
        saved_iter = P_next if save_iters else None
        return State(P_next, lam_next, eta_next, computed_marginals_next, k+1, done), saved_iter


    def skip_updates(state):
        saved_iter = state.P_k if save_iters else None
        return state, saved_iter

    def outer_body_cond(state, _):
        return jax.lax.cond (
            state.done,
            skip_updates,
            one_step,
            state
        )

    # Run the iterations
    (P_final, lam_final, eta_final, _, k, done), iters = jax.lax.scan(
        outer_body_cond,
        init=State(Pk, lam, eta, computed_marginals, 0, False),
        xs=None,
        length=max_outer_iter
    )
    # if done:
    #     jax.debug.print("Algorithm converged at {} total iterations", k)
    # else:
    #     jax.debug.print("Algorithm didn't converge, so stopped after {} total iterations", k)

    return P_final, lam_final, eta_final, iters


def pdhg_algorithm(a: jnp.ndarray, b: jnp.ndarray, C: jnp.ndarray):
    P, *_ = pdhg_quadratic_ot(alpha=a, beta=b, C = C, eps=0)
    return P, jnp.sum(P * C)
