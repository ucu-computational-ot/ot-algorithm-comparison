import numpy as np
from itertools import product
from uot.utils.types import ArrayLike
from typing import List, Tuple
from scipy.special import gamma


import jax.numpy as jnp
from jax import jit


def generate_random_covariance(
    dim: int,
    rng: np.random.Generator,
    diag_linspace: np.ndarray = np.linspace(0.5, 2.0, 10),
    offdiag_linspace: np.ndarray = np.linspace(-0.3, 0.3, 7),
) -> np.ndarray:
    """
    Generates a random symmetric positive definite covariance matrix
    in R^dim, with diagonal entries sampled from diag_linspace and
    off‐diagonals from offdiag_linspace. If the resulting matrix is
    not PD, it adds a small multiple of the identity.
    """
    if dim == 1:
        # For 1D: just pick a variance in diag_linspace
        var = float(rng.choice(diag_linspace, size=1))
        return np.array([[np.round(var, 2)]], dtype=float)

    diag = rng.choice(diag_linspace, size=dim, replace=True)
    cov = np.diag(diag)
    indices = np.triu_indices(dim, k=1)
    for i, j in zip(*indices):
        val = rng.choice(offdiag_linspace)
        cov[i, j] = val
        cov[j, i] = val

    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig <= 0:
        cov += np.eye(dim) * (abs(min_eig) + 1e-6)

    return np.round(cov, 2)


def generate_gmm_coefficients(
    dim: int,
    num_components: int,
    mean_bounds: tuple[float, float],
    rng: np.random.Generator
) -> List[Tuple[Tuple[float, ...], np.ndarray]]:
    """
    Returns a list of length `num_components`, each entry a tuple
    (mean_tuple, covariance_matrix), where:
      - mean_tuple is a length-dim tuple drawn from uniform grid in [-4,4]^dim,
      - covariance_matrix is a random PD matrix in R^dim×R^dim.
    """
    if dim not in [1, 2, 3]:
        raise ValueError("dim must be 1, 2, or 3.")

    # Build a grid of possible means in each coordinate
    mean_range = np.linspace(*mean_bounds, 10)
    if dim == 1:
        all_means = [(float(m),) for m in mean_range]
    else:
        all_means = list(product(mean_range, repeat=dim))

    rng.shuffle(all_means)
    selected_means = all_means[:num_components]

    result: List[Tuple[Tuple[float, ...], np.ndarray]] = []
    for mean in selected_means:
        # TODO: configure the parameters for cov matrix
        cov = generate_random_covariance(dim, rng)
        result.append((tuple(np.round(mean, 2)), cov))
    return result


def get_gmm_pdf(
    dim: int,
    num_components: int,
    mean_bounds: tuple[float, float],
    rng: np.random.Generator,
    use_jax: bool = False,
):
    if dim not in [1, 2, 3]:
        raise ValueError("dim must be 1, 2, or 3.")

    # Step 1: draw random (mean, cov) pairs
    # We reuse the same helper, but seed must be set before shuffling inside generate_coefficients
    params = generate_gmm_coefficients(dim, num_components, mean_bounds, rng)
    means: List[Tuple[float, ...]] = [m for (m, _) in params]
    covs: List[np.ndarray] = [C for (_, C) in params]

    K = num_components
    d = dim

    if use_jax:
        # Convert to JAX arrays
        means_j = jnp.stack([jnp.array(m) for m in means], axis=0)  # (K, d)
        covs_j = jnp.stack([jnp.array(C) for C in covs], axis=0)  # (K, d, d)

        # Precompute inv and det
        inv_j = jnp.linalg.inv(covs_j)  # (K, d, d)
        det_j = jnp.linalg.det(covs_j)  # (K,)
        norm_consts = 1.0 / jnp.sqrt(((2 * jnp.pi) ** d) * det_j)  # (K,)

        def pdf_fn(X: ArrayLike) -> ArrayLike:
            """
            X: jnp.ndarray of shape (N, d).
            Returns: jnp.ndarray of shape (N,) giving mixture density at each row.
            """
            X = jnp.asarray(X)
            if X.ndim != 2 or X.shape[1] != d:
                raise ValueError(f"Input to pdf_fn must be shape (N, {d}).")

            # Broadcast differences: shape (N, K, d)
            diffs = X[:, None, :] - means_j[None, :, :]  # (N, K, d)
            # Quadratic forms: (N,K)
            qf = jnp.einsum("nkd,kde,nke->nk", diffs, inv_j, diffs)
            # (N, K)
            exps = jnp.exp(-0.5 * qf)
            weighted = norm_consts[None, :] * exps  # (N, K)
            return jnp.sum(weighted, axis=1) / K  # (N,)

        return jit(pdf_fn)

    else:
        # NumPy version
        means_np = np.stack(means, axis=0)  # (K, d)
        covs_np = np.stack(covs, axis=0)  # (K, d, d)

        inv_np = np.linalg.inv(covs_np)  # (K, d, d)
        det_np = np.linalg.det(covs_np)  # (K,)
        norm_consts = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_np)  # (K,)

        def pdf_fn(X: ArrayLike) -> ArrayLike:
            """
            X: np.ndarray of shape (N, d).
            Returns: np.ndarray of shape (N,) giving mixture density.
            """
            X = np.asarray(X)
            if X.ndim != 2 or X.shape[1] != d:
                raise ValueError(f"Input to pdf_fn must be shape (N, {d}).")

            # Broadcast diffs: (N, K, d)
            diffs = X[:, None, :] - means_np[None, :, :]  # (N, K, d)
            # Quadratic forms: (N, K)
            qf = np.einsum("nkd,kde,nke->nk", diffs, inv_np, diffs)
            # (N, K)
            exps = np.exp(-0.5 * qf)
            weighted = norm_consts[None, :] * exps  # (N, K)
            # (N,)
            return np.sum(weighted, axis=1) / K

        return pdf_fn


def generate_cauchy_parameters(
    dim: int,
    mean_bounds: tuple[float, float],
    rng: np.random.Generator
):
    if dim not in [1, 2, 3]:
        raise ValueError("dim must be 1, 2, or 3.")

    mean_start, mean_end = mean_bounds

    mean_start = np.full(dim, mean_start)
    mean_end = np.full(dim, mean_end)

    mean = mean_start + rng.uniform() * (mean_end - mean_start)

    return mean, generate_random_covariance(dim=dim, rng=rng)
    

def get_cauchy_pdf(
    dim: int,
    mean_bounds: tuple[float, float],
    rng: np.random.Generator,
    use_jax: bool = False,
):

    mean, cov = generate_cauchy_parameters(dim=dim, mean_bounds=mean_bounds, rng=rng)

    d = dim

    if use_jax:
        cov_inv = jnp.linalg.inv(cov)

        def pdf_fn(X: ArrayLike):
            X = jnp.asarray(X)

            if X.ndim != 2 or X.shape[1] != d:
                raise ValueError(f"Input to pdf_fn must be shape (N, {d}).")

            diff = X - mean

            qf = jnp.einsum("nd,de,ne->n", diff, cov_inv, diff)

            numerator = gamma((d + 1) / 2)
            denominator = (
                gamma(0.5) * (np.pi ** (d / 2)) *
                (jnp.linalg.det(cov) ** 0.5) *
                (1 + qf) ** ((d + 1) / 2)
            )

            return numerator / denominator

    else:
        cov_inv = np.linalg.inv(cov)

        def pdf_fn(X: ArrayLike):
            X = np.asarray(X)

            if X.ndim != 2 or X.shape[1] != d:
                raise ValueError(f"Input to pdf_fn must be shape (N, {d}).")

            diff = X - mean

            qf = np.einsum("nd,de,ne->n", diff, cov_inv, diff)

            numerator = gamma((d + 1) / 2)
            denominator = (
                gamma(0.5) * (np.pi ** (d / 2)) *
                (np.linalg.det(cov) ** 0.5) *
                (1 + qf) ** ((d + 1) / 2)
            )

            return numerator / denominator

    return pdf_fn


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
                raise ValueError(f"Input to pdf_fn must be shape (N, 1).")

            return jnp.where(X >= 0, scale * np.exp(-scale * X), 0)
    
    else:

        def pdf_fn(X: ArrayLike):
            X = np.asarray(X)

            if X.ndim != 2 or X.shape[1] != 1:
                raise ValueError(f"Input to pdf_fn must be shape (N, 1).")

            return np.where(X >= 0, scale * np.exp(-scale * X), 0)

    return pdf_fn

