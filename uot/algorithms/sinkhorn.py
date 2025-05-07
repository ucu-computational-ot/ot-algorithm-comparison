import ot
import jax
import numpy as np
import jax.numpy as jnp
from ott.solvers import linear
from ott.geometry import geometry

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from uot.algorithms.utils import regularize_input

@jax.jit
def coupling_tensor(u: jnp.ndarray,
                    v: jnp.ndarray,
                    cost: jnp.ndarray,
                    epsilon: float) -> jnp.ndarray:
    return jnp.exp((u[:, None] + v[None, :] - cost) / epsilon)

@jax.jit
def tensor_marginals(matrix: jnp.ndarray):
    row_marginal = jnp.sum(matrix, axis=1)      # sum over columns
    column_marginal = jnp.sum(matrix, axis=0)   # sum over rows
    return (row_marginal, column_marginal)

@jax.jit
def compute_error(u: jnp.ndarray,
                  v: jnp.ndarray,
                  a: jnp.ndarray,
                  b: jnp.ndarray,
                  cost: jnp.ndarray,
                  epsilon: float) -> float:
    P = coupling_tensor(u, v, cost, epsilon)
    row_marginal, col_marginal = tensor_marginals(P)
    return jnp.max(
        jnp.array([
            jnp.linalg.norm(a - row_marginal),
            jnp.linalg.norm(b - col_marginal)
        ])
    ) 


@jax.jit
def sinkhorn(a: jnp.ndarray,
             b: jnp.ndarray,
             cost: jnp.ndarray,
             epsilon: float = 1e-3,
             precision: float = 1e-4,
             max_iters: int = 10_000):
    
    n = a.shape[0]
    m = b.shape[0]

    # Initialize dual variables
    u = jnp.zeros(n)
    v = jnp.zeros(m)

    # Define the loop condition
    def cond_fn(carry):
        u_, v_, i_, err_ = carry
        return jnp.logical_and(err_ > precision, i_ < max_iters)

    # Define one iteration body
    def body_fn(carry):
        u_, v_, i_, e = carry

        u_upd = u_ + epsilon * jnp.log(a) - epsilon * logsumexp((u_[:, None] + v_[None, :] - cost) / epsilon, axis=1)
        v_upd = v_ + epsilon * jnp.log(b) - epsilon * logsumexp((u_upd[:, None] + v_[None, :] - cost) / epsilon, axis=0)

        err_upd = jax.lax.cond(
            i_ % 10 == 0,
            lambda: compute_error(u_upd, v_upd, a, b, cost, epsilon),
            lambda: e
        )

        return (u_upd, v_upd, i_ + 1, err_upd)

    # Run the loop
    init_err = compute_error(u, v, a, b, cost, epsilon)
    u_final, v_final, i_final, final_err = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (u, v, jnp.array(0), init_err)
    )

    return u_final, v_final


def jax_sinkhorn(a, b, C, epsilon=1e-3, tolerance = 1e-4):
    a, b, C = regularize_input(a, b, C)
    u, v = sinkhorn(a, b, C, epsilon)
    u.block_until_ready()
    v.block_until_ready()
    P = coupling_tensor(u, v, C, epsilon)

    error = compute_error(u, v, a, b, C, epsilon)
    converged = error < tolerance

    return P, jnp.sum(P * C), converged


def sink(a, b, cost, epsilon=1e-3):
    return linear.solve(
        geometry.Geometry(cost_matrix=cost, epsilon=epsilon),
        a=a,
        b=b,
        lse_mode=True,
        threshold=1e-4
    )

sink_2vmap = jax.jit(sink)

def ott_jax_sinkhorn(mu, nu, C, epsilon=0.001, threshold=1e-4):
    solution = sink_2vmap(mu, nu, C)
    solution.matrix.block_until_ready()
    return solution.matrix, jnp.sum(solution.matrix * C)
