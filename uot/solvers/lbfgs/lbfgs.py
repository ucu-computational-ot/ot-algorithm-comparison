from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxopt import LBFGS, OptStep

from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike
from uot.utils.solver_helpers import coupling_tensor


class LBFGSTwoMarginalSolver(BaseSolver):

    def __init__(self):
        super().__init__()

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
        mu, nu = marginals[0].to_discrete()[1], marginals[1].to_discrete()[1]

        marginals = jnp.array([mu, nu])

        result = lbfgs_multimarginal(
            marginals=marginals,
            C=costs[0],
            epsilon=reg,
            maxiter=maxiter,
            tolerance=tol
        )

        potentials = result.params
        u, v = potentials[0][None, :], potentials[1][:, None]

        transport_plan = coupling_tensor(u, v, costs[0], epsilon=reg)

        return {
            "transport_plan": transport_plan,
            "cost": (transport_plan * costs[0]).sum(), 
            "u_final": u,
            "v_final": v,
            "iterations": result.state.iter_num,
            "error": result.state.error,
        }

    
@jax.jit
def lbfgs_multimarginal(marginals: jnp.ndarray,
             C: jnp.ndarray,
             epsilon: float = 1,
             tolerance: float = 1e-4,
             maxiter: int = 10000) -> OptStep:

    N = marginals.shape[0]
    n = marginals.shape[1]

    shapes = [tuple(n if j == i else 1 for j in range(N)) for i in range(N)]
    potentials = jnp.zeros_like(marginals)

    @jax.jit
    def objective(potentials: jax.Array):
        """Computes the dual objective with logsumexp stabilization."""
        potentials_reshaped = [potentials[i].reshape(shapes[i]) for i in range(N)]
        potentials_sum = sum(potentials_reshaped)
        log_sub_entropy = (potentials_sum - C) / epsilon
        max_log_sub_entropy = jnp.max(log_sub_entropy, axis=0, keepdims=True)
        stable_sum = jnp.exp(max_log_sub_entropy) * jnp.sum(
            jnp.exp(log_sub_entropy - max_log_sub_entropy), axis=0
        )
        dual = potentials * marginals
        return -jnp.sum(dual - epsilon * stable_sum)

    solver = LBFGS(fun=objective, tol=tolerance, maxiter=maxiter, maxls=100)
    result = solver.run(init_params=potentials)
    
    return result

