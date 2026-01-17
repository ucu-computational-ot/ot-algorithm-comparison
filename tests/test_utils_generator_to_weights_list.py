import numpy as np
import jax.numpy as jnp

from uot.data.measure import DiscreteMeasure
from uot.problems.problem_generator import ProblemGenerator
from uot.problems.two_marginal import TwoMarginalProblem
from uot.utils.generator_to_weights_list import generator_to_weights_list


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
    assert len(weights_list) == 1
    weights = weights_list[0]
    np.testing.assert_allclose(np.asarray(weights[0]), weights_mu)
    np.testing.assert_allclose(np.asarray(weights[1]), weights_nu)


def test_generator_to_weights_list_excludes_zeros():
    points = np.array([[0.0], [1.0], [2.0]])
    weights_mu = np.array([0.2, 0.0, 0.8])
    weights_nu = np.array([0.5, 0.0, 0.5])
    generator = DummyGenerator([_make_problem(points, weights_mu, weights_nu)])

    support, weights_list = generator_to_weights_list(generator, include_zeros=False)

    np.testing.assert_allclose(np.asarray(support), points[[0, 2]])
    weights = weights_list[0]
    np.testing.assert_allclose(np.asarray(weights[0]), weights_mu[[0, 2]])
    np.testing.assert_allclose(np.asarray(weights[1]), weights_nu[[0, 2]])


def test_generator_to_weights_list_multiple_problems():
    points = np.array([[0.0], [1.0]])
    generator = DummyGenerator([
        _make_problem(points, np.array([1.0, 0.0]), np.array([0.5, 0.5])),
        _make_problem(points, np.array([0.3, 0.7]), np.array([0.0, 1.0])),
    ])

    support, weights_list = generator_to_weights_list(generator, include_zeros=True)

    np.testing.assert_allclose(np.asarray(support), points)
    assert len(weights_list) == 2
    weights_0 = weights_list[0]
    weights_1 = weights_list[1]
    np.testing.assert_allclose(np.asarray(weights_0[1]), np.array([0.5, 0.5]))
    np.testing.assert_allclose(np.asarray(weights_1[0]), np.array([0.3, 0.7]))
