import ot
import jax
import numpy as np
import jax.numpy as jnp
from ott.solvers import linear
from ott.geometry import geometry
from ott.geometry import pointcloud

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

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
             reg: float = 1e-3,
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

        u_upd = u_ + reg * jnp.log(a) - reg * logsumexp((u_[:, None] + v_[None, :] - cost) / reg, axis=1)
        v_upd = v_ + reg * jnp.log(b) - reg * logsumexp((u_upd[:, None] + v_[None, :] - cost) / reg, axis=0)

        err_upd = jax.lax.cond(
            i_ % 10 == 0,
            lambda: compute_error(u_upd, v_upd, a, b, cost, reg),
            lambda: e
        )

        return (u_upd, v_upd, i_ + 1, err_upd)

    # Run the loop
    init_err = compute_error(u, v, a, b, cost, reg)
    u_final, v_final, i_final, final_err = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (u, v, jnp.array(0), init_err)
    )

    P = coupling_tensor(u_final, v_final, cost, reg)
    return P

def jax_sinkhorn(a, b, C, epsilon=1e-3):
    P = sinkhorn(a, b, C).block_until_ready()
    return P, jnp.sum(P * C)

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

def pot_sinkhorn(a, b, C, epsilon=0.001):
    P = ot.sinkhorn(a, b, C, reg=epsilon, stopThr=1e-4)
    return P, np.sum(P * C)
