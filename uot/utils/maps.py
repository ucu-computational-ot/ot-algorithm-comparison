import jax.numpy as jnp

from uot.utils.types import ArrayLike


def barycentric_map_from_plan(
    plan: ArrayLike,
    target_points: ArrayLike,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Compute the barycentric projection (a map) induced by a transport plan.

    Args:
        plan: transport plan of shape (n_source, n_target)
        target_points: coordinates of the target support (n_target, d)
        eps: numerical stability constant to avoid division by zero

    Returns:
        Array of shape (n_source, d) with the barycentric image of each source bin.
    """
    P = jnp.asarray(plan)
    Y = jnp.asarray(target_points)
    if P.ndim != 2:
        raise ValueError(f"plan must be 2-D, got shape {P.shape}")
    if Y.ndim != 2:
        raise ValueError(f"target_points must be 2-D, got shape {Y.shape}")
    weights = jnp.maximum(jnp.sum(P, axis=1, keepdims=True), eps)
    return (P @ Y) / weights


def mean_conditional_variance(
    plan: ArrayLike,
    target_points: ArrayLike,
    a: ArrayLike | None = None,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Compute the (weighted) mean conditional variance of a transport plan.

    Args:
        plan: transport plan (n_source, n_target)
        target_points: coordinates of the target support (n_target, d)
        a: optional source marginal weights (n_source,)
        eps: numerical stability constant

    Returns:
        Scalar variance proxy measuring map diffuseness (0 for deterministic maps).
    """
    P = jnp.asarray(plan)
    Y = jnp.asarray(target_points)
    if a is None:
        a = jnp.sum(P, axis=1)
    else:
        a = jnp.asarray(a)
    Ey = (P @ Y) / jnp.maximum(a[:, None], eps)
    Ey2 = (P @ jnp.sum(Y * Y, axis=1)) / jnp.maximum(a, eps)
    var_i = Ey2 - jnp.sum(Ey * Ey, axis=1)
    return jnp.sum(a * var_i)
