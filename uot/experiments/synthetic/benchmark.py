from uot.problems.generators import GaussianMixtureGenerator
from uot.utils.costs import cost_euclid_squared


gaussian_mixture = GaussianMixtureGenerator(
    name="Gaussians (1000pts)",
    dim=1,
    num_components=1,
    n_points=2000,
    num_problems=5,
    borders=(-6, 6),
    cost_fn=cost_euclid_squared,
    use_jax=True,
    seed=43,
)

gaussian_mixture.generate()