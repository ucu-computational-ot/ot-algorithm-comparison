import numpy as np
import jax.numpy as jnp
import pytest

from uot.utils.generate_nd_grid import generate_nd_grid


def cartesian_product(axes):
    """
    Reference implementation using pure Python for testing.
    """
    if not axes:
        return np.zeros((0, 0))
    grids = np.meshgrid(*axes, indexing='ij')
    stacked = np.stack(grids, axis=-1)
    return stacked.reshape(-1, len(axes))


@pytest.mark.parametrize("axes, use_jax", [
    ([np.array([0, 1])], False),
    ([np.array([0, 1])], True),
    ([np.array([0, 1, 2]), np.array([10, 20])], False),
    ([np.array([0, 1, 2]), np.array([10, 20])], True),
    ([np.array([0]), np.array([1]), np.array([2])], False),
    ([np.array([0]), np.array([1]), np.array([2])], True),
])
def test_generate_nd_grid_shapes_and_values(axes, use_jax):
    # Compute grid with function under test
    grid = generate_nd_grid(axes, use_jax=use_jax)

    # Convert jax arrays to numpy for comparison
    if use_jax:
        grid_np = np.asarray(grid)
    else:
        grid_np = grid

    # Reference result
    ref = cartesian_product([np.asarray(a) for a in axes])

    # Check shape
    assert grid_np.shape == ref.shape, f"Shape mismatch: {grid_np.shape} vs {ref.shape}"

    # Check contents
    np.testing.assert_array_equal(grid_np, ref)


def test_empty_axes_list():
    # Edge case: empty axes list -> should return array of shape (1, 0)
    grid = generate_nd_grid([], use_jax=False)
    assert grid.shape == (1, 0)

    grid_jax = generate_nd_grid([], use_jax=True)
    # Convert to numpy
    grid_jnp = np.asarray(grid_jax)
    assert grid_jnp.shape == (1, 0)


def test_dtype_consistency():
    # Ensure output dtype matches input dtype
    axes = [np.array([0, 1], dtype=np.float32),
            np.array([10, 20], dtype=np.int32)]
    grid = generate_nd_grid(axes, use_jax=False)
    # float and int promote to default numpy dtype
    assert grid.dtype == np.result_type(*[a.dtype for a in axes])

    grid_jax = generate_nd_grid(axes, use_jax=True)
    # jax dtype
    assert grid_jax.dtype == jnp.result_type(
        *[jnp.asarray(a).dtype for a in axes])


def test_jit_compilation():
    # Basic smoke test: calling twice yields same result and uses cached compilation
    axes = [jnp.arange(3), jnp.linspace(0, 1, 2)]
    def fn(): return generate_nd_grid(axes, use_jax=True)
    # First call compiles
    result1 = fn()
    # Second call should be fast and produce same
    result2 = fn()
    np.testing.assert_array_equal(np.asarray(result1), np.asarray(result2))
