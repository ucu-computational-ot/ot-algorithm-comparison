from typing import List, Callable, Iterator
import numpy as np
import jax.numpy as jnp
from numpy.random import default_rng
from uot.problems.problem_generator import ProblemGenerator
from uot.problems.two_marginal import TwoMarginalProblem
from uot.data.measure import DiscreteMeasure
from uot.utils.types import ArrayLike
from uot.utils.generate_nd_grid import generate_nd_grid
from uot.utils.generator_helpers import get_cauchy_pdf


class CauchyGenerator(ProblemGenerator):

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
        super().__init__()
        # TODO: arbitrary dim?
        if dim not in [1]:
            raise ValueError("dim must be 1, 2 or 3")

        self._name = name
        self._dim = dim
        self._n_points = n_points
        self._num_datasets = num_datasets
        self._borders = borders
        self._cost_fn = cost_fn
        self._use_jax = use_jax
        self._rng = default_rng(seed)

    def generate(self) -> Iterator[TwoMarginalProblem]:
        pdfs_num = 2 * self._num_datasets
        axes_support = self._get_axes(self._n_points)
        mean_bounds = (self._borders[0] * 0.9, self._borders[1] * 0.9)
        points = generate_nd_grid(axes_support)
        cauchy_pdfs = [
            get_cauchy_pdf(
                dim=self._dim,
                mean_bounds=mean_bounds,
                rng=self._rng,
                use_jax=self._use_jax,
            )
            for _ in range(pdfs_num)
        ]

        for i in range(self._num_datasets):
            mu_weights = cauchy_pdfs[2 * i](points)
            mu_weights /= mu_weights.sum()
            nu_weights = cauchy_pdfs[2 * i + 1](points)
            nu_weights /= nu_weights.sum()
            mu = DiscreteMeasure(points=points, weights=mu_weights)
            nu = DiscreteMeasure(points=points, weights=nu_weights)
            yield TwoMarginalProblem(
                name=self._name,
                mu=mu,
                nu=nu,
                cost_fn=self._cost_fn,
            )

    def _get_axes(self, n_points: int) -> List[ArrayLike]:
        lib = jnp if self._use_jax else np
        ax = lib.linspace(self._borders[0], self._borders[1], n_points)
        axs = [ax for _ in range(self._dim)]
        return axs

