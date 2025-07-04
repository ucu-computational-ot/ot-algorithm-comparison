import numpy as np
import jax.numpy as jnp
import pytest
from uot.utils.generate_nd_grid import generate_nd_grid
from uot.utils.generator_helpers import get_exponential_pdf, get_axes

rng = np.random.default_rng(0)


def exp_pdf_np(x, scale):
    return np.where(x >= 0, scale * np.exp(-scale * x), 0)


@pytest.mark.parametrize("scale_bounds,use_jax", [
    ((1.0, 1.0), False),
    ((2.0, 2.0), False),
    ((1.0, 1.0), True),
    ((0.5, 0.5), True),
])
def test_pdf_values_exact(scale_bounds, use_jax):
    # When scale_bounds are equal, scale is deterministic
    pdf_fn = get_exponential_pdf(scale_bounds, rng, use_jax=use_jax)
    scale = scale_bounds[0]

    # Use get_axes and generate_nd_grid to build test points
    # Create a grid spanning from -1 to 3 with 5 points
    axes_support = get_axes(1, (-1.0, 3.0), 5, use_jax=use_jax)
    points = generate_nd_grid(axes_support, use_jax=use_jax)  # shape (5,1)

    # Evaluate PDF
    if use_jax:
        vals = pdf_fn(points)
        vals = np.asarray(vals)
    else:
        vals = pdf_fn(points)

    # Analytical values
    expected = exp_pdf_np(points, scale)
    np.testing.assert_allclose(vals, expected, rtol=1e-6, atol=0)
    np.testing.assert_allclose(vals, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("use_jax", [False, True])
def test_invalid_shape_error(use_jax):
    pdf_fn = get_exponential_pdf((1.0, 2.0), rng, use_jax=use_jax)
    # 1D array
    arr1 = np.array([0.0, 1.0])
    # wrong second dim
    arr2 = np.zeros((5, 2))
    for bad in [arr1, arr2]:
        bad_input = jnp.array(bad) if use_jax else bad
        with pytest.raises(ValueError):
            pdf_fn(bad_input)
