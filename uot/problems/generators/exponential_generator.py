from collections.abc import Callable, Iterator
from numpy.random import default_rng

from uot.data.measure import DiscreteMeasure
from uot.problems.two_marginal import TwoMarginalProblem
from uot.problems.problem_generator import ProblemGenerator

from uot.utils.types import ArrayLike
from uot.utils.generate_nd_grid import generate_nd_grid, compute_cell_volume
from uot.utils.generator_helpers import get_exponential_pdf, get_axes
from uot.utils.build_measure import _build_measure


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
        measure_mode: str = "grid",  # NEW: 'grid' | 'discrete' | 'auto'
        cell_discretization: str = "cell-centered" # NEW: 'cell-centered' | 'vertex-centered'
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
        self._measure_mode = measure_mode
        self.cell_discretization = cell_discretization

    def generate(self, *args, **kwargs) -> Iterator[TwoMarginalProblem]:
        pdfs_num = 2 * self._num_datasets
        axes_support = get_axes(
            dim=self._dim,
            borders=self._borders,
            n_points=self._n_points,
            cell_discretization=self.cell_discretization,
            use_jax=self._use_jax,
        )
        scale_bounds = (0.1, self._borders[1] * 0.5)
        points = generate_nd_grid(axes_support, use_jax=self._use_jax)
        cell_volume = compute_cell_volume(axes_support, use_jax=self._use_jax)

        def _prepare(weights):
            if self.cell_discretization == "cell-centered":
                weights = weights * cell_volume
            return weights / weights.sum()
        exponential_pdfs = [
            get_exponential_pdf(
                scale_bounds=scale_bounds,
                rng=self._rng,
                use_jax=self._use_jax,
            )
            for _ in range(pdfs_num)
        ]

        for i in range(self._num_datasets):
            # NOTE: for some reason the returned shape is still (n, 1)
            #       so we just reshape it to be back (n,)
            mu_weights = exponential_pdfs[2 * i](points).reshape(-1)
            mu_weights = _prepare(mu_weights)
            nu_weights = exponential_pdfs[2 * i + 1](points).reshape(-1)
            nu_weights = _prepare(nu_weights)
            # mu = DiscreteMeasure(points=points, weights=mu_weights)
            # nu = DiscreteMeasure(points=points, weights=nu_weights)
            mu = _build_measure(points, mu_weights, axes_support, self._measure_mode, self._use_jax)
            nu = _build_measure(points, nu_weights, axes_support, self._measure_mode, self._use_jax)

            yield TwoMarginalProblem(
                name=self._name,
                mu=mu,
                nu=nu,
                cost_fn=self._cost_fn,
            )
