import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cholesky, solve_triangular
from scipy.stats import wishart
from collections.abc import Callable

# ——— Constants ———
GAUSSIAN_MEAN_SAMPLE_GRID_N = 100
GAUSSIAN_VARIANCE_SAMPLE_GRID_N = 100
GAUSSIAN_MEAN_PRECISION = 4
GAUSSIAN_VARIANCE_PRECISION = 4
DEFAULT_JITTER = 1e-6

PRNGKey = jax.random.PRNGKey


def generate_random_covariance(
    key: PRNGKey,
    dim: int,
    diag_linspace: jnp.ndarray = None,
    offdiag_linspace: jnp.ndarray = None,
) -> tuple[PRNGKey, jnp.ndarray]:
    """
    JAX-only: generate a random SPD covariance in R^dim.
    Returns new key and a (dim,dim) SPD matrix.
    """
    if diag_linspace is None:
        diag_linspace = jnp.linspace(0.5, 2.0, GAUSSIAN_VARIANCE_SAMPLE_GRID_N)
    if offdiag_linspace is None:
        offdiag_linspace = jnp.linspace(-0.3,
                                        0.3, GAUSSIAN_VARIANCE_SAMPLE_GRID_N)

    if dim == 1:
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (), 0, diag_linspace.shape[0])
        var = diag_linspace[idx]
        return key, var.reshape(1, 1)

    # sample diagonals and off-diagonals
    key, k1, k2 = jax.random.split(key, 3)
    idxs = jax.random.randint(k1, (dim,), 0, diag_linspace.shape[0])
    diag = diag_linspace[idxs]
    mat = jnp.diag(diag)
    iu = jnp.triu_indices(dim, k=1)
    idx_off = jax.random.randint(
        k2, (iu[0].shape[0],), 0, offdiag_linspace.shape[0])
    offvals = offdiag_linspace[idx_off]
    mat = mat.at[iu].set(offvals)
    mat = mat.at[(iu[1], iu[0])].set(offvals)

    # ensure PD by jitter until cholesky succeeds
    def try_chol(A, jit_count):
        def _chol(a): return cholesky(a, lower=True)
        try:
            return _chol(A), A
        except Exception:
            jitter = (DEFAULT_JITTER * (10**jit_count))
            A2 = A + jnp.eye(dim) * jitter
            return try_chol(A2, jit_count+1)
    L, A_pd = try_chol(mat, 0)
    return key, A_pd


def generate_gmm_coefficients(
    key: PRNGKey,
    dim: int,
    num_components: int,
    mean_bounds: tuple[float, float],
    variance_bounds: tuple[float, float],
) -> tuple[PRNGKey, jnp.ndarray, jnp.ndarray]:
    """
    Returns key, means (K,d), covs (K,d,d)
    """
    # sample means
    low, high = mean_bounds
    grid = jnp.linspace(low, high, GAUSSIAN_MEAN_SAMPLE_GRID_N)

    def sample_mean(k):
        ks = jax.random.split(k, dim+1)
        coords = [grid[jax.random.randint(
            ks[i], (), 0, grid.shape[0])] for i in range(1, dim+1)]
        return ks[0], jnp.round(jnp.stack(coords), GAUSSIAN_MEAN_PRECISION)
    key, *mean_keys = jax.random.split(key, num_components+1)
    _, means = jax.lax.scan(lambda carry, mk: sample_mean(
        mk), key, jnp.array(mean_keys))
    means = means.reshape(num_components, dim)

    # prepare linspaces
    if variance_bounds[0] > 0:
        diag_space = jnp.geomspace(
            variance_bounds[0],
            variance_bounds[1],
            GAUSSIAN_VARIANCE_SAMPLE_GRID_N,
        )
    else:
        diag_space = jnp.linspace(
            variance_bounds[0],
            variance_bounds[1],
            GAUSSIAN_VARIANCE_SAMPLE_GRID_N,
        )
    half_off = variance_bounds[1]/10.0
    off_space = jnp.linspace(-half_off, half_off,
                             GAUSSIAN_VARIANCE_SAMPLE_GRID_N)

    # sample covariances
    def gen_cov(carry, _):
        k = carry
        k, cov = generate_random_covariance(k, dim, diag_space, off_space)
        return k, cov
    key, covs = jax.lax.scan(gen_cov, key, None, length=num_components)
    covs = covs.reshape(num_components, dim, dim)

    return key, means, covs


def build_gmm_pdf(
    means: jnp.ndarray,
    covs: jnp.ndarray,
    weights: jnp.ndarray = None
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Pure-JAX GMM PDF builder, returning jitted function.
    """
    K, d = means.shape
    if weights is None:
        weights = jnp.ones((K,)) / K
    log_w = jnp.log(weights)

    # batched cholesky
    Ls = vmap(lambda C: cholesky(C, lower=True))(covs)  # (K,d,d)
    log_dets = 2.0 * \
        jnp.sum(jnp.log(jnp.diagonal(Ls, axis1=1, axis2=2)), axis=1)
    log_norm = -0.5*(d*jnp.log(2*jnp.pi) + log_dets) + log_w

    @jit
    def pdf_fn(X: jnp.ndarray) -> jnp.ndarray:
        # X: (N,d)
        diffs = X[:, None, :] - means[None, :, :]         # (N,K,d)
        # solve for each k
        diffs_T = diffs.transpose(1, 2, 0)
        y = vmap(lambda L, dm: solve_triangular(L, dm, lower=True),
                 in_axes=(0, 0))(Ls, diffs_T)
        y = y.transpose(2, 0, 1)                           # (N,K,d)
        qf = jnp.sum(y**2, axis=2)                       # (N,K)
        log_p = log_norm[None, :] - 0.5 * qf                 # (N,K)
        m = jnp.max(log_p, axis=1, keepdims=True)
        s = jnp.sum(jnp.exp(log_p - m), axis=1)
        return jnp.exp(m.squeeze() + jnp.log(s))

    return pdf_fn


def get_gmm_pdf(
    key: PRNGKey,
    dim: int,
    num_components: int,
    mean_bounds: tuple[float, float],
    variance_bounds: tuple[float, float]
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], PRNGKey]:
    """
    Convenience: sample GMM params and return PDF and updated key.
    """
    key, means, covs = generate_gmm_coefficients(
        key, dim, num_components, mean_bounds, variance_bounds
    )
    pdf = build_gmm_pdf(means, covs)
    return pdf, key


def build_gmm_pdf_scipy(
    means: np.ndarray,      # shape (K, d)
    covs: np.ndarray,       # shape (K, d, d)
    weights: np.ndarray     # shape (K,), must sum to 1
):
    """
    Returns a function pdf(X: np.ndarray) -> np.ndarray of shape (N,), where
    each row of X is evaluated under the mixture.
    """
    K, d = means.shape
    weights = weights / np.sum(weights)

    # Precompute per-component inverse covariances and normalization constants.
    chol = np.linalg.cholesky(covs)  # (K, d, d)
    log_dets = 2.0 * np.sum(
        np.log(np.diagonal(chol, axis1=1, axis2=2)), axis=1
    )
    log_norm = -0.5 * (d * np.log(2 * np.pi) + log_dets) + np.log(weights)
    eye = np.eye(d)
    inv_chol = np.linalg.solve(chol, eye)  # (K, d, d)
    inv_cov = np.matmul(inv_chol.transpose(0, 2, 1), inv_chol)  # (K, d, d)

    def pdf_np(X: np.ndarray) -> np.ndarray:
        diffs = X[:, None, :] - means[None, :, :]  # (N, K, d)
        qf = np.einsum("nkd,kde,nke->nk", diffs, inv_cov, diffs)  # (N, K)
        log_p = log_norm[None, :] - 0.5 * qf
        m = np.max(log_p, axis=1, keepdims=True)
        return np.exp(m[:, 0]) * np.sum(np.exp(log_p - m), axis=1)

    return pdf_np


def sample_gmm_params_wishart(
    K: int,
    d: int,
    mean_bounds: tuple[float, float],
    wishart_df: int,
    wishart_scale: np.ndarray,
    rng: np.random.Generator
):
    # means uniform
    means = rng.uniform(mean_bounds[0], mean_bounds[1], size=(K, d))

    # covariances Wishart
    covs = wishart.rvs(df=wishart_df, scale=wishart_scale,
                       size=K, random_state=rng)
    # ensure it is an array and K is present if K=1
    covs = np.array(covs)
    if covs.ndim == 0:
        covs = covs.reshape(1, 1, 1)
    if covs.ndim == 1:
        covs = covs[:, None, None]
    if covs.ndim == 2:
        covs = covs[np.newaxis, :]

    weights = np.ones(K) / K
    return means, covs, weights
