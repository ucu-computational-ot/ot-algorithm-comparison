import jax
from jax import numpy as jnp
from jax import lax, random
import optax

from collections.abc import Sequence
from functools import partial
from uot.solvers.base_solver import BaseSolver
from uot.data.measure import DiscreteMeasure


@partial(jax.jit, static_argnums=(0, 5, 6))
def _saga(
    self,
    mu: jnp.ndarray,        # shape (n,)
    nu: jnp.ndarray,        # shape (m,)
    C: jnp.ndarray,         # shape (n, m)
    reg: float,
    maxiter: int,
    tol: float,
    key: jnp.ndarray,       # PRNGKey
):
    maxiter = int(maxiter)
    n, m = mu.shape[0], nu.shape[0]
    size = n + m
    # init state
    theta0 = jnp.zeros((size,))    # [phi; psi]
    mem0 = jnp.zeros((size,))    # gradient memory table
    avg0 = 0.0                    # average of mem
    err0 = jnp.inf
    state0 = (0, theta0, mem0, avg0, err0, key)

    def cond_fn(state):
        i, theta, mem, avg_mem, error, key = state
        return (i < maxiter) & (error > tol)

    def step(state):
        i, theta, mem, avg_mem, error, key = state

        # sample index k
        key, sub = random.split(key)
        k = random.randint(sub, (), 0, size)
        # unpack theta
        phi = theta[:n]
        psi = theta[n:]
        log_K = (phi[:, None] + psi[None, :] - C) / reg

        # compute gradient at k
        def grad_at_k(k):

            def for_phi(_):
                row = log_K[k, :]
                E1 = jnp.exp(row - jnp.max(row))
                return mu[k] - E1.sum()

            def for_psi(_):
                j = k - n
                col = log_K[:, j]
                E2 = jnp.exp(col - jnp.max(col))
                return nu[j] - E2.sum()

            return lax.cond(k < n, for_phi, for_psi, operand=None)

        gk = grad_at_k(k)
        old_mem_k = mem[k]
        vk = gk - old_mem_k + avg_mem
        # update step size from schedule
        eta = self.schedule(i)
        theta = theta.at[k].add(eta * vk)
        # update mem and avg
        mem = mem.at[k].set(gk)
        avg_mem = avg_mem + (gk - old_mem_k) / size

        # periodic full error computation
        def compute_err(_):
            phi1 = theta[:n]
            psi1 = theta[n:]
            P = jnp.exp((phi1[:, None] + psi1[None, :] - C) / reg)
            return jnp.maximum(
                jnp.linalg.norm(P.sum(axis=1) - mu),
                jnp.linalg.norm(P.sum(axis=0) - nu),
            )
        error = lax.cond((i % 10) == 0, compute_err, lambda _: error, operand=None)
        return (i + 1, theta, mem, avg_mem, error, key)

    # run loop
    i_final, theta_final, mem_final, avg_final, err_final, _ = lax.while_loop(
        cond_fn, step, state0
    )
    # unpack results
    phi_final = theta_final[:n]
    psi_final = theta_final[n:]
    P = jnp.exp((phi_final[:, None] + psi_final[None, :] - C) / reg)
    cost = jnp.sum(P * C)
    return P, phi_final, psi_final, cost, err_final, i_final


class SAGASolver(BaseSolver):
    def __init__(
        self,
        learning_rate: float = 0.003,
        seed: int = 0,
    ):
        super().__init__()
        self.schedule = optax.linear_schedule(
            init_value=learning_rate*10,
            end_value=learning_rate,
            transition_steps=20_000,
        )
        self._seed = seed

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[jnp.ndarray],
        reg: float = 1e-3,
        maxiter: int = 1000,
        tol: float = 1e-6,
        *args,
        **kwargs,
    ):
        (mu, nu) = (
            marginals[0].to_discrete(include_zeros=False)[1],
            marginals[1].to_discrete(include_zeros=False)[1]
            )
        C = costs[0]
        key = jax.random.PRNGKey(self._seed)
        P, phi, psi, cost, error, iters = _saga(
            self, mu, nu, C, reg, maxiter, tol, key
        )
        return {
            "transport_plan": P,
            "cost":           cost,
            "u_final":        phi,
            "v_final":        psi,
            "iterations":     iters,
            "error":          error,
        }
