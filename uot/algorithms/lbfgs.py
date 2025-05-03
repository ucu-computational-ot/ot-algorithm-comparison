import jax
import numpy as np
import jax.numpy as jnp
from ot.smooth import smooth_ot_dual
from jaxopt import LBFGS

def dual_lbfgs(a, b, C, epsillon=1e-3):
    P = smooth_ot_dual(a, b, C, reg=epsillon, reg_type="negentropy")
    return P, np.sum(P * C)


def dual_lbfs_potentials(a, b, C, epsilon=1e-3):
    _, log = smooth_ot_dual(a, b, C, reg=epsilon, reg_type="negentropy",
                            log=True)
    return log['alpha'], log['beta']
    

@jax.jit
def lbfgs_ot(marginals: jnp.ndarray,
             C: jnp.ndarray,
             epsilon: float):

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
        return jnp.sum(dual - epsilon * stable_sum)

    solver = LBFGS(fun=objective)
    result = solver.run(init_params=potentials)

    print(result.params)