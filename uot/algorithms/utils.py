import jax.numpy as jnp


def regularize_input(a: jnp.ndarray, 
                     b: jnp.ndarray, 
                     C: jnp.ndarray, 
                     max_mass_add: float = 1e-30):

    a_zero_size = jnp.sum(a == 0)
    a = a.at[a == 0].set(max_mass_add / a_zero_size)

    b_zero_size = jnp.sum(b == 0)
    b = b.at[b == 0].set(max_mass_add / b_zero_size)

    C_zero_size = jnp.sum(C == 0)
    C = C.at[C == 0].set(max_mass_add / C_zero_size)

    return a, b, C




