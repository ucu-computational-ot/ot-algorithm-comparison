import numpy as np
import jax.numpy as jnp


from uot.utils.types import ArrayLike


def get_exponential_pdf(
    scale_bounds: tuple[float, float],
    rng: np.random.Generator,
    use_jax: bool = False,
):

    scale_start, scale_end = scale_bounds
    scale = scale_start + rng.uniform() * (scale_end - scale_start)

    if use_jax:

        def pdf_fn(X: ArrayLike):
            X = jnp.asarray(X)

            if X.ndim != 2 or X.shape[1] != 1:
                raise ValueError("Input to pdf_fn must be shape (N, 1).")

            return jnp.where(X >= 0, scale * np.exp(-scale * X), 0)

    else:

        def pdf_fn(X: ArrayLike):
            X = np.asarray(X)

            if X.ndim != 2 or X.shape[1] != 1:
                raise ValueError("Input to pdf_fn must be shape (N, 1).")

            return np.where(X >= 0, scale * np.exp(-scale * X), 0)

    return pdf_fn
