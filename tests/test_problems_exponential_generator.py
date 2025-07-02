import numpy as np
import pytest

from uot.utils.generate_nd_grid import generate_nd_grid
from uot.utils.generator_helpers import get_axes
from uot.problems.generators.exponential_generator import ExponentialGenerator
from uot.data.measure import DiscreteMeasure
from uot.problems.two_marginal import TwoMarginalProblem

# Dummy cost function: Euclidean distance
cost_fn = lambda x, y: np.linalg.norm(x - y, axis=1)

@pytest.mark.parametrize("use_jax", [False, True])
def test_generate_raises_on_dimension(use_jax):
    # ExponentialGenerator only supports dim=1
    with pytest.raises(ValueError):
        ExponentialGenerator(
            name="exp", dim=2, n_points=5, num_datasets=1,
            borders=(0.0, 1.0), cost_fn=cost_fn, use_jax=use_jax, seed=0
        )

@pytest.mark.parametrize("use_jax,num_datasets", [(False, 2), (True, 3)])
def test_generate_returns_problems(use_jax, num_datasets):
    dim = 1
    n_points = 10
    borders = (0.0, 5.0)
    rng_seed = 123

    gen = ExponentialGenerator(
        name="exp", dim=dim, n_points=n_points,
        num_datasets=num_datasets, borders=borders,
        cost_fn=cost_fn, use_jax=use_jax, seed=rng_seed
    )

    # Precompute support grid
    axes = get_axes(dim, borders, n_points, use_jax=use_jax)
    points = generate_nd_grid(axes, use_jax=use_jax)
    M = n_points
    expected_count = num_datasets

    problems = list(gen.generate())
    assert len(problems) == expected_count

    for prob in problems:
        assert isinstance(prob, TwoMarginalProblem)
        # Check names
        assert prob.name == "exp"
        # Check marginals
        mu, nu = prob._mu, prob._nu
        assert isinstance(mu, DiscreteMeasure)
        assert isinstance(nu, DiscreteMeasure)
        # Points match support
        pmu, wmu = mu.to_discrete()
        pnu, wnu = nu.to_discrete()
        assert pmu.shape == (M, 1)
        # Weights sum to 1
        assert wmu.shape == (M, 1)
        assert wnu.shape == (M, 1)
        np.testing.assert_allclose(wmu.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(wnu.sum(), 1.0, atol=1e-6)
