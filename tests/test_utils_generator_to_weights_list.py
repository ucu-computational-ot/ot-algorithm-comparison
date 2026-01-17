import numpy as np
import jax.numpy as jnp

from uot.data.measure import DiscreteMeasure, GridMeasure
from uot.problems.problem_generator import ProblemGenerator
from uot.problems.two_marginal import TwoMarginalProblem
from uot.utils.generator_to_weights_list import (
    generator_to_weights_list,
    generator_to_weights_array,
)


class DummyGenerator(ProblemGenerator):
    def __init__(self, problems):
        self._problems = problems

    def generate(self):
        for problem in self._problems:
            yield problem


def _make_problem(points, weights_mu, weights_nu):
    mu = DiscreteMeasure(points=points, weights=weights_mu)
    nu = DiscreteMeasure(points=points, weights=weights_nu)
    return TwoMarginalProblem(
        name="dummy",
        mu=mu,
        nu=nu,
        cost_fn=lambda x, y: jnp.zeros((x.shape[0], y.shape[0])),
    )


def test_generator_to_weights_list_includes_zeros():
    points = np.array([[0.0], [1.0], [2.0]])
    weights_mu = np.array([0.2, 0.0, 0.8])
    weights_nu = np.array([0.0, 1.0, 0.0])
    generator = DummyGenerator([_make_problem(points, weights_mu, weights_nu)])

    support, weights_list = generator_to_weights_list(generator, include_zeros=True)

    np.testing.assert_allclose(np.asarray(support), points)
    assert len(weights_list) == 2
    np.testing.assert_allclose(np.asarray(weights_list[0]), weights_mu)
    np.testing.assert_allclose(np.asarray(weights_list[1]), weights_nu)


def test_generator_to_weights_list_excludes_zeros():
    points = np.array([[0.0], [1.0], [2.0]])
    weights_mu = np.array([0.2, 0.0, 0.8])
    weights_nu = np.array([0.5, 0.0, 0.5])
    generator = DummyGenerator([_make_problem(points, weights_mu, weights_nu)])

    support, weights_list = generator_to_weights_list(generator, include_zeros=False)

    np.testing.assert_allclose(np.asarray(support), points[[0, 2]])
    assert len(weights_list) == 2
    np.testing.assert_allclose(np.asarray(weights_list[0]), weights_mu[[0, 2]])
    np.testing.assert_allclose(np.asarray(weights_list[1]), weights_nu[[0, 2]])


def test_generator_to_weights_list_multiple_problems():
    points = np.array([[0.0], [1.0]])
    generator = DummyGenerator([
        _make_problem(points, np.array([1.0, 0.0]), np.array([0.5, 0.5])),
        _make_problem(points, np.array([0.3, 0.7]), np.array([0.0, 1.0])),
    ])

    support, weights_list = generator_to_weights_list(generator, include_zeros=True)

    np.testing.assert_allclose(np.asarray(support), points)
    assert len(weights_list) == 4
    np.testing.assert_allclose(np.asarray(weights_list[1]), np.array([0.5, 0.5]))
    np.testing.assert_allclose(np.asarray(weights_list[2]), np.array([0.3, 0.7]))


def test_generator_to_weights_list_2d_discrete_support():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    weights_mu = np.array([0.5, 0.0, 0.5])
    weights_nu = np.array([0.2, 0.0, 0.8])
    generator = DummyGenerator([_make_problem(points, weights_mu, weights_nu)])

    support, weights_list = generator_to_weights_list(generator, include_zeros=False)

    np.testing.assert_allclose(np.asarray(support), points[[0, 2]])
    assert len(weights_list) == 2
    np.testing.assert_allclose(np.asarray(weights_list[0]), weights_mu[[0, 2]])
    np.testing.assert_allclose(np.asarray(weights_list[1]), weights_nu[[0, 2]])


def test_generator_to_weights_list_grid_mode():
    axes = [np.array([0.0, 1.0]), np.array([10.0, 20.0, 30.0])]
    weights_mu = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.3]])
    weights_nu = np.array([[0.3, 0.0, 0.1], [0.2, 0.1, 0.0]])
    mu = GridMeasure(axes=axes, weights_nd=weights_mu, normalize=False)
    nu = GridMeasure(axes=axes, weights_nd=weights_nu, normalize=False)
    problem = TwoMarginalProblem(
        name="grid",
        mu=mu,
        nu=nu,
        cost_fn=lambda x, y: jnp.zeros((x.shape[0], y.shape[0])),
    )
    generator = DummyGenerator([problem])
    generator._measure_mode = "grid"

    support, weights_list = generator_to_weights_list(generator, include_zeros=False)

    assert support.shape == (2, 3, 2)
    np.testing.assert_allclose(np.asarray(support[0, 0]), np.array([0.0, 10.0]))
    np.testing.assert_allclose(np.asarray(support[1, 2]), np.array([1.0, 30.0]))
    assert len(weights_list) == 2
    np.testing.assert_allclose(np.asarray(weights_list[0]), weights_mu)
    np.testing.assert_allclose(np.asarray(weights_list[1]), weights_nu)


def test_generator_to_weights_list_grid_mode_2d_more_points():
    axes = [np.linspace(-1.0, 1.0, 4), np.linspace(0.0, 3.0, 5)]
    weights_mu = np.arange(20).reshape(4, 5).astype(float)
    weights_nu = np.flip(weights_mu, axis=1)
    mu = GridMeasure(axes=axes, weights_nd=weights_mu, normalize=False)
    nu = GridMeasure(axes=axes, weights_nd=weights_nu, normalize=False)
    problem = TwoMarginalProblem(
        name="grid-2d",
        mu=mu,
        nu=nu,
        cost_fn=lambda x, y: jnp.zeros((x.shape[0], y.shape[0])),
    )
    generator = DummyGenerator([problem])
    generator._measure_mode = "grid"

    support, weights_list = generator_to_weights_list(generator, include_zeros=False)

    assert support.shape == (4, 5, 2)
    np.testing.assert_allclose(np.asarray(support[0, 0]), np.array([-1.0, 0.0]))
    np.testing.assert_allclose(np.asarray(support[3, 4]), np.array([1.0, 3.0]))
    assert len(weights_list) == 2
    np.testing.assert_allclose(np.asarray(weights_list[0]), weights_mu)
    np.testing.assert_allclose(np.asarray(weights_list[1]), weights_nu)


def test_generator_to_weights_array_discrete():
    points = np.array([[0.0], [1.0], [2.0]])
    weights_mu = np.array([0.2, 0.3, 0.5])
    weights_nu = np.array([0.1, 0.6, 0.3])
    generator = DummyGenerator([_make_problem(points, weights_mu, weights_nu)])

    support, weights_array = generator_to_weights_array(generator, include_zeros=True)

    np.testing.assert_allclose(np.asarray(support), points)
    assert weights_array.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(weights_array[0, :]), weights_mu)
    np.testing.assert_allclose(np.asarray(weights_array[1, :]), weights_nu)


def test_generator_to_weights_array_grid_mode():
    axes = [np.linspace(-1.0, 1.0, 4), np.linspace(0.0, 3.0, 5)]
    weights_mu = np.arange(20).reshape(4, 5).astype(float)
    weights_nu = np.flip(weights_mu, axis=0)
    mu = GridMeasure(axes=axes, weights_nd=weights_mu, normalize=False)
    nu = GridMeasure(axes=axes, weights_nd=weights_nu, normalize=False)
    problem = TwoMarginalProblem(
        name="grid-2d",
        mu=mu,
        nu=nu,
        cost_fn=lambda x, y: jnp.zeros((x.shape[0], y.shape[0])),
    )
    generator = DummyGenerator([problem])
    generator._measure_mode = "grid"

    support, weights_array = generator_to_weights_array(generator, include_zeros=True)

    assert support.shape == (4, 5, 2)
    assert weights_array.shape == (2, 4, 5)
    np.testing.assert_allclose(np.asarray(weights_array[0, :, :]), weights_mu)
    np.testing.assert_allclose(np.asarray(weights_array[1, :, :]), weights_nu)
