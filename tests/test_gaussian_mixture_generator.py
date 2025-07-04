import numpy as np

from uot.problems.generators.gaussian_mixture_generator import GaussianMixtureGenerator
from uot.utils.costs import cost_euclid_squared


def test_gaussian_mixture_produces_distinct_datasets():
    gen = GaussianMixtureGenerator(
        name="",
        dim=1,
        num_components=2,
        n_points=10,
        num_datasets=2,
        borders=(-10, 10),
        cost_fn=cost_euclid_squared,
        use_jax=False,
        seed=52,
    )

    problems = gen.generate()
    problem0_marginals = problems[0].get_marginals()
    problem1_marginals = problems[1].get_marginals()
    # compare mu and nu separately
    assert not np.allclose(
        problem0_marginals[0].to_discrete()[1],
        problem1_marginals[0].to_discrete()[1],
    )
    assert not np.allclose(
        problem0_marginals[1].to_discrete()[1],
        problem1_marginals[1].to_discrete()[1]
    )
