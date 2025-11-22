from collections.abc import Iterator, Callable

import numpy as np
from scipy.stats import cauchy

from uot.utils.generate_nd_grid import generate_nd_grid, compute_cell_volume
from uot.utils.generator_helpers import get_axes
from uot.data.measure import DiscreteMeasure
from uot.problems.two_marginal import TwoMarginalProblem
from uot.problems.problem_generator import ProblemGenerator
from uot.utils.build_measure import _build_measure

MEAN_FROM_BORDERS_COEF = 0.9
VAR_LOWER = 0.05
VAR_UPPER = 0.5


class IndependentCauchyGenerator(ProblemGenerator):
    """
    Two-marginal problems with independent Cauchy marginals,
    where mu and nu each have their own location & scale parameters.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        n_points: int,
        num_datasets: int,
        borders: tuple[float, float],
        cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        seed: int = 42,
        measure_mode: str = "grid",  # NEW: 'grid' | 'discrete' | 'auto'
        cell_discretization: str = "cell-centered" # NEW: 'cell-centered' | 'vertex-centered'
    ):
        super().__init__()
        self._name = name
        self._dim = dim
        self._n_points = n_points
        self._num_datasets = num_datasets
        self._borders = borders
        self._cost_fn = cost_fn
        self._rng = np.random.default_rng(seed)
        self._measure_mode = measure_mode
        self._use_jax = False
        self.cell_discretization = cell_discretization

    def generate(self) -> Iterator[TwoMarginalProblem]:
        # build the evaluation grid once
        axes = get_axes(
            self._dim,
            self._borders,
            self._n_points,
            cell_discretization=self.cell_discretization,
            use_jax=False,
        )
        points = generate_nd_grid(axes, use_jax=False)
        cell_volume = compute_cell_volume(axes, use_jax=False)

        def _prepare(weights: np.ndarray) -> np.ndarray:
            if self.cell_discretization == "cell-centered":
                weights = weights * cell_volume
            return weights / weights.sum()

        mean_bounds = (
            self._borders[0] * MEAN_FROM_BORDERS_COEF,
            self._borders[1] * MEAN_FROM_BORDERS_COEF,
        )
        scale_bounds = (
            abs(self._borders[1]) * VAR_LOWER,
            abs(self._borders[1]) * VAR_UPPER,
        )

        for _ in range(self._num_datasets):
            # --- sample loc & scale for mu marginal ---
            locs_mu = self._rng.uniform(mean_bounds[0], mean_bounds[1], size=self._dim)
            scales_mu = self._rng.uniform(scale_bounds[0], scale_bounds[1], size=self._dim)
            pdf_mu = np.prod([
                cauchy(loc=locs_mu[i], scale=scales_mu[i]).pdf(points[:, i])
                for i in range(self._dim)
            ], axis=0)
            w_mu = _prepare(pdf_mu)

            # --- sample loc & scale for nu marginal ---
            locs_nu = self._rng.uniform(mean_bounds[0], mean_bounds[1], size=self._dim)
            scales_nu = self._rng.uniform(scale_bounds[0], scale_bounds[1], size=self._dim)
            pdf_nu = np.prod([
                cauchy(loc=locs_nu[i], scale=scales_nu[i]).pdf(points[:, i])
                for i in range(self._dim)
            ], axis=0)
            w_nu = _prepare(pdf_nu)

            # mu_measure = DiscreteMeasure(points=points, weights=w_mu)
            # nu_measure = DiscreteMeasure(points=points, weights=w_nu)
            mu_measure = _build_measure(points, w_mu, axes, self._measure_mode, self._use_jax)
            nu_measure = _build_measure(points, w_nu, axes, self._measure_mode, self._use_jax)

            yield TwoMarginalProblem(
                name=self._name,
                mu=mu_measure,
                nu=nu_measure,
                cost_fn=self._cost_fn,
            )
