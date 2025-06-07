import numpy as np
from itertools import product
from uot.utils.types import ArrayLike
from typing import Callable, List, Tuple


try:
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    jit = None
    JAX_AVAILABLE = False


def generate_random_covariance(
    dim: int,
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
        var = float(np.random.choice(diag_linspace, size=1))
        return np.array([[np.round(var, 2)]], dtype=float)

    diag = np.random.choice(diag_linspace, size=dim, replace=True)
    cov = np.diag(diag)
    indices = np.triu_indices(dim, k=1)
    for i, j in zip(*indices):
        val = np.random.choice(offdiag_linspace)
        cov[i, j] = val
        cov[j, i] = val

    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig <= 0:
        cov += np.eye(dim) * (abs(min_eig) + 1e-6)

    return np.round(cov, 2)


def generate_coefficients(dim: int, distributions: dict[str, int]) -> dict[str, list]:
    """
    Generates random parameters for several 1D/2D/3D distributions.
    For 'gaussian', returns a list of (mean_tuple, covariance_matrix) pairs.
    For others (gamma, beta, uniform, cauchy, white-noise), returns
    a list of parameter tuples (rounded to 2 decimals).
    """
    if dim not in [1, 2, 3]:
        raise ValueError("dim must be 1, 2, or 3.")

    # TODO: these parameters should not be defined here (hardcoded ones are bad)
    basic_ranges = {
        "mean_range": (-4, 4),
        "std_range": (0.3, 1.5),
        "shape_range": (1.0, 3.0),
        "scale_range": (0.2, 1.5),
        "loc_range": (-3.5, 3.5),
        "alpha_range": (0.5, 5.0),
        "beta_range": (0.5, 5.0),
        "width_range": (3.0, 6.0),
        "lower_range": (-5.0, 0.0),
    }

    distribution_parameters = {
        "gaussian": ("mean", "std"),
        "gamma": ("shape", "scale"),
        "beta": ("alpha", "beta"),
        "uniform": ("lower", "width"),
        "cauchy": ("loc", "scale"),
        "white-noise": ("mean", "std"),
    }

    results: dict[str, list] = {}

    for distribution, num_to_generate in distributions.items():
        if distribution not in distribution_parameters:
            raise ValueError(f"Unsupported distribution: {distribution}")

        param_names = distribution_parameters[distribution]
        param_ranges: list[list] = []

        for param in param_names:
            if f"{param}_range" not in basic_ranges:
                raise ValueError(f"Missing range for {param}.")
            values = np.linspace(*basic_ranges[f"{param}_range"], 10)
            if dim == 1:
                param_ranges.append(values.tolist())
            else:
                # All combinations of param in R^dim
                product_vals = list(product(values, repeat=dim))
                param_ranges.append(product_vals)

        if distribution == "gaussian" and dim > 1:
            # Build mean choices in R^dim
            mean_choices = list(
                product(np.linspace(
                    *basic_ranges["mean_range"], 10), repeat=dim)
            )
            np.random.shuffle(mean_choices)
            selected_means = mean_choices[:num_to_generate]
            result: list[tuple[tuple[float, ...], np.ndarray]] = []
            for mean in selected_means:
                cov = generate_random_covariance(dim)
                result.append((tuple(np.round(mean, 2)), cov))
            results[distribution] = result

        else:
            all_combinations = list(product(*param_ranges))
            np.random.shuffle(all_combinations)
            selected_combinations = all_combinations[:num_to_generate]
            rounded_combinations = [
                tuple(np.round(combination, 2)) for combination in selected_combinations
            ]
            results[distribution] = rounded_combinations

    return results


def generate_gmm_coefficients(
    dim: int, num_components: int
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
    mean_range = np.linspace(-4.0, 4.0, 10)
    if dim == 1:
        all_means = [(float(m),) for m in mean_range]
    else:
        all_means = list(product(mean_range, repeat=dim))

    np.random.shuffle(all_means)
    selected_means = all_means[:num_components]

    result: List[Tuple[Tuple[float, ...], np.ndarray]] = []
    for mean in selected_means:
        cov = generate_random_covariance(dim)
        result.append((tuple(np.round(mean, 2)), cov))
    return result


def get_gmm_pdf(dim: int, num_components: int, use_jax: bool = False, seed: int = 0):
    if dim not in [1, 2, 3]:
        raise ValueError("dim must be 1, 2, or 3.")

    # Step 1: draw random (mean, cov) pairs
    rng = np.random.default_rng(seed)
    # We reuse the same helper, but seed must be set before shuffling inside generate_coefficients
    np.random.seed(seed)
    params = generate_gmm_coefficients(dim, num_components)
    means: List[Tuple[float, ...]] = [m for (m, _) in params]
    covs: List[np.ndarray] = [C for (_, C) in params]

    K = num_components
    d = dim

    if use_jax and JAX_AVAILABLE:
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

            N = X.shape[0]
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
