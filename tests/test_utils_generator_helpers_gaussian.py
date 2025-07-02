import numpy as np
import jax.numpy as jnp
from jax import random
import pytest
from scipy.stats import multivariate_normal

# Import the pure-JAX GMM utilities
from uot.utils.generator_helpers.gaussian import (
    generate_random_covariance,
    generate_gmm_coefficients,
    build_gmm_pdf,
    get_gmm_pdf,
)

# Seed key
key = random.PRNGKey(0)

# --- Tests for generate_random_covariance ---


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_generate_random_covariance_sym_pd(dim):
    """
    Test that JAX-generated covariance is symmetric and positive-definite.
    """
    new_key, cov = generate_random_covariance(key, dim)
    # cov should be (dim, dim)
    assert cov.shape == (dim, dim)
    # symmetry
    cov_np = np.asarray(cov)
    assert np.allclose(cov_np, cov_np.T)
    # positive definiteness
    eigs = np.linalg.eigvalsh(cov_np)
    assert np.all(eigs > 0)

# --- Tests for generate_gmm_coefficients ---


def test_generate_gmm_coefficients_shapes_and_bounds():
    dim, K = 2, 4
    mean_bounds = (-2.0, 2.0)
    var_bounds = (0.5, 1.5)
    new_key, means, covs = generate_gmm_coefficients(
        key, dim, K, mean_bounds, var_bounds)
    # shapes
    assert means.shape == (K, dim)
    assert covs.shape == (K, dim, dim)
    # means within bounds
    means_np = np.asarray(means)
    assert means_np.min() >= mean_bounds[0]
    assert means_np.max() <= mean_bounds[1]
    # covariances PD
    for i in range(K):
        cov_np = np.asarray(covs[i])
        assert np.allclose(cov_np, cov_np.T)
        eigs = np.linalg.eigvalsh(cov_np)
        assert np.all(eigs > 0)

# --- Tests for build_gmm_pdf ---


def test_build_gmm_pdf_single_component():
    # 1D Gaussian with variance=3
    means = jnp.array([[0.0]])
    covs = jnp.array([[[3.0]]])
    pdf = build_gmm_pdf(means, covs)
    xs = jnp.array([[0.0], [1.0], [2.0]])
    vals = pdf(xs)
    # reference values
    sigma2 = 3.0
    ref = (1.0/np.sqrt(2*np.pi*sigma2)) * \
        np.exp(-np.array([0.0, 1.0, 4.0])/(2*sigma2))
    np.testing.assert_allclose(np.asarray(vals), ref, rtol=1e-6)


def test_build_gmm_pdf_mixture():
    # two-component 1D mixture
    means = jnp.array([[-1.0], [1.0]])
    covs = jnp.tile(jnp.eye(1)[None], (2, 1, 1))
    pdf = build_gmm_pdf(means, covs)
    xs = jnp.linspace(-3, 3, 7).reshape(-1, 1)
    vals = np.asarray(pdf(xs))
    # reference: average of two normals
    ref = 0.5 * multivariate_normal(-1, 1).pdf(xs.flatten()) + \
        0.5 * multivariate_normal(1, 1).pdf(xs.flatten())
    np.testing.assert_allclose(vals, ref, rtol=1e-6)

# --- Tests for get_gmm_pdf ---


def test_get_gmm_pdf_end_to_end():
    dim, K = 1, 3
    mean_bounds = (-1.0, 1.0)
    var_bounds = (0.2, 0.8)
    pdf, new_key = get_gmm_pdf(key, dim, K, mean_bounds, var_bounds)
    # pdf is callable
    assert callable(pdf)
    xs = jnp.linspace(-1, 1, 5).reshape(-1, 1)
    vals = pdf(xs)
    # correct shape
    assert vals.shape == (5,)
    # non-negative
    assert jnp.all(vals >= 0)
    # key updated
    assert not jnp.array_equal(new_key, key)
