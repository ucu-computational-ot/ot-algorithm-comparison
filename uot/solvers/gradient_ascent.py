from collections.abc import Sequence

import jax
import jax.numpy as jnp
import optax

from uot.utils.types import ArrayLike
from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver

from uot.utils.solver_helpers import coupling_tensor


class GradientAscentTwoMarginalSolver(BaseSolver):
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
        if len(costs) == 0:
            raise ValueError("Cost tensors not defined.")
        if len(marginals) != 2:
            raise ValueError("This gradient ascent solver accepts only two marginals.")
        mu, nu = marginals[0].to_discrete()[1], marginals[1].to_discrete()[1]

        marginals = jnp.array([mu, nu])

        final_potentials, i_final, final_loss, final_err = gradient_ascent_opt_multimarginal(
            marginals=marginals,
            cost=costs[0],
            eps=reg,
            tol=tol,
            max_iterations=maxiter,
        )

        u, v = final_potentials[0][None, :].reshape(-1), final_potentials[1][:, None].reshape(-1)

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
def gradient_ascent_opt_multimarginal(
    marginals,
    cost,
    eps=1e-3,
    learning_rate=1e-3,
    max_iterations=100_000,
    tol=1e-4,
) -> jax.Array:
    N = marginals.shape[0]
    n = marginals.shape[1]

    shapes = [tuple(n if j == i else 1 for j in range(N)) for i in range(N)]
    potentials = jnp.zeros_like(marginals)
    optimizer = optax.sgd(learning_rate=learning_rate)
    opt_state = optimizer.init(potentials)

    @jax.jit
    def objective(potentials: jax.Array):
        """Computes the dual objective with logsumexp stabilization."""
        potentials_reshaped = [potentials[i].reshape(shapes[i]) for i in range(N)]
        potentials_sum = sum(potentials_reshaped)
        log_sub_entropy = (potentials_sum - cost) / eps
        max_log_sub_entropy = jnp.max(log_sub_entropy, axis=0, keepdims=True)
        stable_sum = jnp.exp(max_log_sub_entropy) * jnp.sum(
            jnp.exp(log_sub_entropy - max_log_sub_entropy), axis=0
        )
        dual = potentials * marginals
        return jnp.sum(dual - eps * stable_sum)

    objective_gradient = jax.value_and_grad(objective)

    @jax.jit
    def step(state: tuple[int, jax.Array, optax.OptState, float, float, bool]):
        """Performs one gradient ascent step."""
        i, potentials, opt_state, prev_loss, prev_err, _ = state
        loss, grad = objective_gradient(potentials)
        # minus gradient because we are performing gradient ascent
        updates, opt_state = optimizer.update(-grad, opt_state, potentials)
        potentials = optax.apply_updates(potentials, updates)
        # L-infinity norm
        max_change = jnp.max(jnp.abs(potentials - state[1]))
        has_converged = max_change < tol
        return i + 1, potentials, opt_state, loss, max_change, has_converged

    def cond_fn(
            state: tuple[int, jax.Array, optax.OptState, float, float, bool]
    ):
        i, _, _, _, _, has_converged = state
        return jnp.logical_and(
            i < max_iterations,
            jnp.logical_not(has_converged),
        )

    final_state = jax.lax.while_loop(
        cond_fn, step, (0, potentials, opt_state, jnp.inf, jnp.inf, False)
    )

    steps, final_potentials, _, final_loss, final_err, has_converged = final_state
    return final_potentials, steps, final_loss, final_err
