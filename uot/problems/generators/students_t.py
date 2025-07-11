from collections.abc import Iterator, Callable

import numpy as np
from scipy.stats import multivariate_t

from uot.utils.generate_nd_grid import generate_nd_grid
from uot.utils.generator_helpers import sample_gmm_params_wishart, get_axes
from uot.data.measure import DiscreteMeasure
from uot.problems.two_marginal import TwoMarginalProblem
from uot.problems.problem_generator import ProblemGenerator

MEAN_FROM_BORDERS_COEF = 0.9
VAR_LOWER = 0.05
VAR_UPPER = 0.5


class StudentTGenerator(ProblemGenerator):
    """
    Two-marginal problems where each marginal is a multivariate Student's t
    with its own randomly drawn mean & covariance.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        nu: float,
        n_points: int,
        num_datasets: int,
        borders: tuple[float, float],
        cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        seed: int = 42,
    ):
        super().__init__()
        self._name = name
        self._dim = dim
        self._nu = nu
        self._n_points = n_points
        self._num_datasets = num_datasets
        self._borders = borders
        self._cost_fn = cost_fn
        self._rng = np.random.default_rng(seed)
        # Wishart params for covariance sampling
        self._wishart_df = dim + 1
        self._wishart_scale = np.eye(dim)

    def generate(self) -> Iterator[TwoMarginalProblem]:
        # build the evaluation grid once
        axes = get_axes(self._dim, self._borders, self._n_points, use_jax=False)
        points = generate_nd_grid(axes, use_jax=False)

        mean_bounds = (
            self._borders[0] * MEAN_FROM_BORDERS_COEF,
            self._borders[1] * MEAN_FROM_BORDERS_COEF,
        )
        variance_bounds = (
            abs(self._borders[1]) * VAR_LOWER,
            abs(self._borders[1]) * VAR_UPPER,
        )

        for _ in range(self._num_datasets):
            # --- sample parameters for mu marginal ---
            mus1, covs1, _ = sample_gmm_params_wishart(
                K=1,
                d=self._dim,
                mean_bounds=mean_bounds,
                wishart_df=self._wishart_df,
                wishart_scale=self._wishart_scale,
                rng=self._rng,
            )
            rv_mu = multivariate_t(loc=mus1[0], shape=covs1[0], df=self._nu)
            w_mu = rv_mu.pdf(points)
            w_mu /= w_mu.sum()

            # --- sample parameters for nu marginal (independent) ---
            mus2, covs2, _ = sample_gmm_params_wishart(
                K=1,
                d=self._dim,
                mean_bounds=mean_bounds,
                wishart_df=self._wishart_df,
                wishart_scale=self._wishart_scale,
                rng=self._rng,
            )
            rv_nu = multivariate_t(loc=mus2[0], shape=covs2[0], df=self._nu)
            w_nu = rv_nu.pdf(points)
            w_nu /= w_nu.sum()

            mu_measure = DiscreteMeasure(points=points, weights=w_mu)
            nu_measure = DiscreteMeasure(points=points, weights=w_nu)

            yield TwoMarginalProblem(
                name=self._name,
                mu=mu_measure,
                nu=nu_measure,
                cost_fn=self._cost_fn,
            )
