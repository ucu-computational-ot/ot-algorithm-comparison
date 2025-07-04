import jax
import jax.numpy as jnp

@jax.jit
def coupling_tensor(
    u: jnp.ndarray, v: jnp.ndarray, cost: jnp.ndarray, epsilon: float
) -> jnp.ndarray:
    return jnp.exp((u[:, None] + v[None, :] - cost) / epsilon)


@jax.jit
def tensor_marginals(matrix: jnp.ndarray):
    row_marginal = jnp.sum(matrix, axis=1)  # sum over columns
    column_marginal = jnp.sum(matrix, axis=0)  # sum over rows
    return (row_marginal, column_marginal)