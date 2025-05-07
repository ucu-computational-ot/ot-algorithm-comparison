import jax
import numpy as np
import jax.numpy as jnp
from jaxopt import LBFGS

    
@jax.jit
def lbfgs_multimarginal(marginals: jnp.ndarray,
             C: jnp.ndarray,
             epsilon: float = 1,
             tolerance: float = 1e-4):

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

    solver = LBFGS(fun=objective, tol=tolerance)
    result = solver.run(init_params=potentials)

    return result.params, result.state.error < tolerance

def lbfgs_ot(a: jnp.ndarray, 
             b: jnp.ndarray,
             C: jnp.ndarray,
             tolerance: float = 1e-4, 
             epsilon: float = 1e-3):
    marginals = jnp.array([a, b])
    potentials, converged = lbfgs_multimarginal(marginals=marginals,
                                                C = C,
                                                tolerance=tolerance,
                                                epsilon=epsilon)

    P = jnp.exp(
        (potentials[0][None, :] + potentials[1][:, None] - C) / epsilon
    )

    return P, jnp.sum(P * C), converged