from collections.abc import Iterator, Callable

import numpy as np
from scipy.stats import expon

from uot.utils.generate_nd_grid import generate_nd_grid, compute_cell_volume
from uot.utils.generator_helpers import get_axes
from uot.data.measure import DiscreteMeasure
from uot.problems.two_marginal import TwoMarginalProblem
from uot.problems.problem_generator import ProblemGenerator
from uot.utils.build_measure import _build_measure

from uot.utils.logging import setup_logger

logger = setup_logger(__name__)

MEAN_LOWER_FROM_BORDERS_COEF = 0.05
MEAN_UPPER_FROM_BORDERS_COEF = 0.3


class IndependentExponentialGenerator(ProblemGenerator):
    """
    Two-marginal problems with independent exponential marginals,
    where each marginal has its own loc & scale per dimension.
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
        self._use_jax = False
        self._cost_fn = cost_fn
        self._rng = np.random.default_rng(seed)
        self._measure_mode = measure_mode
        self.cell_discretization = cell_discretization

    def _sample_exponential_weights(self, points: np.ndarray) -> np.ndarray:
        """
        Sample independent loc & scale for each dimension, compute the
        product-PDF on `points`, and return a normalized weight vector.
        """
        # We choose loc_i within the grid bounds so support covers negatives if any.
        span = abs(self._borders[1] - self._borders[0])
        locs = self._rng.uniform(
            0,
            span * 0.25,
            size=self._dim,
        )
        # Scale must be positive; we pick it relative to border span.
        scales = self._rng.uniform(
            2 * MEAN_LOWER_FROM_BORDERS_COEF / span,
            2 * MEAN_UPPER_FROM_BORDERS_COEF / span,
            size=self._dim,
        )

        # Compute product of independent exponential PDFs
        pdf_vals = np.prod([
            expon(loc=locs[i], scale=scales[i]).pdf(points[:, i])
            for i in range(self._dim)
        ], axis=0)

        if np.any(np.isnan(pdf_vals)):
            logger.warning('pdf values contain nan')

        # Avoid division by zero: add tiny eps if needed
        total = pdf_vals.sum()
        if total == 0:
            total = np.finfo(float).eps
        return pdf_vals / total

    def generate(self) -> Iterator[TwoMarginalProblem]:
        # build grid once
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
            total = weights.sum()
            if total == 0:
                total = np.finfo(float).eps
            return weights / total

        for _ in range(self._num_datasets):
            # independently sample weights for mu and nu
            w_mu = _prepare(self._sample_exponential_weights(points))
            if np.any(np.isnan(w_mu)):
                logger.warning("w_mu contains nan")
            w_nu = _prepare(self._sample_exponential_weights(points))
            if np.any(np.isnan(w_nu)):
                logger.warning("w_nu contains nan") 

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
