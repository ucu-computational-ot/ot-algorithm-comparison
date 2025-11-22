from collections.abc import Iterator, Callable

import numpy as np
from scipy.stats import genhyperbolic

from uot.utils.generate_nd_grid import generate_nd_grid, compute_cell_volume
from uot.utils.generator_helpers import get_axes
from uot.data.measure import DiscreteMeasure
from uot.problems.two_marginal import TwoMarginalProblem
from uot.problems.problem_generator import ProblemGenerator
from uot.utils.build_measure import _build_measure

MEAN_FROM_BORDERS_COEF = 0.5
VAR_LOWER = 0.05
VAR_UPPER = 0.3


class GeneralizedHyperbolicMixtureGenerator(ProblemGenerator):
    """
    Two-marginal problems with mixtures of independent generalized hyperbolic marginals,
    sampling separate mixture parameters for mu and nu.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        num_components: int,
        n_points: int,
        num_datasets: int,
        borders: tuple[float, float],
        cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        seed: int = 42,
        lambda_bounds: tuple[float, float] = (-1.0, 2.0),
        alpha_bounds: tuple[float, float] = (0.5, 5.0),
        beta_coef: float = 0.9,
        delta_bounds: tuple[float, float] = (0.1, 2.0),
        measure_mode: str = "grid",  # NEW: 'grid' | 'discrete' | 'auto'
        cell_discretization: str = "cell-centered" # NEW: 'cell-centered' | 'vertex-centered'
    ):
        super().__init__()
        self._name = name
        self._dim = dim
        self._num_components = num_components
        self._n_points = n_points
        self._num_datasets = num_datasets
        self._borders = borders
        self._cost_fn = cost_fn
        self._use_jax = False
        self._rng = np.random.default_rng(seed)
        self._lambda_bounds = lambda_bounds
        self._alpha_bounds = alpha_bounds
        self._beta_coef = beta_coef
        self._delta_bounds = delta_bounds
        self._measure_mode = measure_mode
        self.cell_discretization = cell_discretization

    def _sample_mixture_weights_and_pdfs(self, points: np.ndarray) -> np.ndarray:
        """Sample GH mixture parameters and return normalized pdf values on points."""
        # equal mixture weights
        mix_weights = np.ones(self._num_components) / self._num_components

        # sample component parameters
        lambdas = self._rng.uniform(*self._lambda_bounds, size=self._num_components)
        alphas = self._rng.uniform(*self._alpha_bounds, size=self._num_components)
        betas = np.array([
            self._rng.uniform(-abs(a)*self._beta_coef, abs(a)*self._beta_coef)
            for a in alphas
        ])
        deltas = self._rng.uniform(*self._delta_bounds, size=self._num_components)
        locs = self._rng.uniform(
            self._borders[0]*MEAN_FROM_BORDERS_COEF,
            self._borders[1]*MEAN_FROM_BORDERS_COEF,
            size=(self._num_components, self._dim),
        )

        pdf_vals = np.zeros(points.shape[0])
        for i in range(self._num_components):
            # independent across dimensions
            comp_pdf = np.ones(points.shape[0])
            for d in range(self._dim):
                rv = genhyperbolic(
                    p=lambdas[i],
                    a=alphas[i],
                    b=betas[i],
                    loc=locs[i, d],
                    scale=deltas[i],
                )
                comp_pdf *= rv.pdf(points[:, d])
            pdf_vals += mix_weights[i] * comp_pdf

        return pdf_vals / pdf_vals.sum()

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
            return weights / weights.sum()

        for _ in range(self._num_datasets):
            # sample for mu
            w_mu = _prepare(self._sample_mixture_weights_and_pdfs(points))
            # sample independently for nu
            w_nu = _prepare(self._sample_mixture_weights_and_pdfs(points))

            # mu_measure = DiscreteMeasure(points=points, weights=w_mu)
            # nu_measure = DiscreteMeasure(points=points, weights=w_nu)
            mu = _build_measure(points, w_mu, axes, self._measure_mode, self._use_jax)
            nu = _build_measure(points, w_nu, axes, self._measure_mode, self._use_jax)

            yield TwoMarginalProblem(
                name=self._name,
                mu=mu,
                nu=nu,
                cost_fn=self._cost_fn,
            )
