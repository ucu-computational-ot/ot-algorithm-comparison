from collections.abc import Callable, Iterator
from numpy.random import default_rng

from uot.data.measure import DiscreteMeasure
from uot.problems.two_marginal import TwoMarginalProblem
from uot.problems.problem_generator import ProblemGenerator

from uot.utils.types import ArrayLike
from uot.utils.generate_nd_grid import generate_nd_grid
from uot.utils.generator_helpers import get_exponential_pdf, get_axes


class ExponentialGenerator(ProblemGenerator):

    def __init__(
        self,
        name: str,
        dim: int,
        n_points: int,
        num_datasets: int,
        borders: tuple[float, float],
        cost_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
        use_jax: bool = False,
        seed: int = 42,
    ):
        if dim != 1:
            raise ValueError("For exponential distribution dim must be 1")

        self._name = name
        self._dim = dim
        self._n_points = n_points
        self._num_datasets = num_datasets
        self._borders = borders
        self._cost_fn = cost_fn
        self._use_jax = use_jax
        self._rng = default_rng(seed)

    def generate(self, *args, **kwargs) -> Iterator[TwoMarginalProblem]:
        pdfs_num = 2 * self._num_datasets
        axes_support = get_axes(dim=self._dim, borders=self._borders, n_points=self._n_points,
                                use_jax=self._use_jax)
        scale_bounds = (0.1, self._borders[1] * 0.5)
        points = generate_nd_grid(axes_support)
        exponential_pdfs = [
            get_exponential_pdf(
                scale_bounds=scale_bounds,
                rng=self._rng,
                use_jax=self._use_jax,
            )
            for _ in range(pdfs_num)
        ]

        for i in range(self._num_datasets):
            mu_weights = exponential_pdfs[2 * i](points)
            mu_weights /= mu_weights.sum()
            nu_weights = exponential_pdfs[2 * i + 1](points)
            nu_weights /= nu_weights.sum()
            mu = DiscreteMeasure(points=points, weights=mu_weights)
            nu = DiscreteMeasure(points=points, weights=nu_weights)
            yield TwoMarginalProblem(
                name=self._name,
                mu=mu,
                nu=nu,
                cost_fn=self._cost_fn,
            )
