import numpy as np
from uot.utils.types import ArrayLike
from uot.problems.problem_generator import ProblemGenerator
from uot.utils.generate_nd_grid import generate_nd_grid
from uot.utils.generator_helpers import (
    get_gmm_pdf as get_gmm_pdf_jax,
    build_gmm_pdf_scipy,
    sample_gmm_params_wishart
)
from uot.utils.build_measure import _build_measure
from uot.utils.generator_helpers import get_axes
from uot.problems.two_marginal import TwoMarginalProblem
from uot.data.measure import DiscreteMeasure
from collections.abc import Callable, Iterator
import jax
import jax.numpy as jnp


MEAN_FROM_BORDERS_COEF = 0.5
VARIANCE_LOWER_BOUND_COEF = 0.001
VARIANCE_UPPER_BOUND_COEF = 0.5


class GaussianMixtureGenerator(ProblemGenerator):
    """
    Generator of two-marginal problems with GMM marginals on a fixed grid.
    Supports both JAX- and NumPy/SciPy-based pdf evaluation.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        num_components: int,
        n_points: int,
        num_datasets: int,
        borders: tuple[float, float],
        cost_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
        use_jax: bool = True,
        seed: int = 42,
        wishart_df: int | None = None,
        wishart_scale: np.ndarray | None = None,
        measure_mode: str = "grid",  # NEW: 'grid' | 'discrete' | 'auto'
        cell_discretization: str = "cell-centered" # NEW: 'cell-centered' | 'vertex-centered'
    ):
        super().__init__()
        # TODO: arbitrary dim?
        if dim not in [1, 2, 3]:
            raise ValueError("dim must be 1, 2 or 3")
        self._name = name
        self._dim = dim
        self._num_components = num_components
        self._n_points = n_points
        self._num_datasets = num_datasets
        self._borders = borders
        self._cost_fn = cost_fn
        self._use_jax = use_jax
        self._measure_mode = measure_mode
        # Wishart parameters for NumPy path
        # default df is dim+1 if not provided, scale defaults to identity
        self._wishart_df = wishart_df if wishart_df is not None else dim + 1
        self._wishart_scale = wishart_scale if wishart_scale is not None else np.eye(
            dim)
        if self._use_jax:
            self._key = jax.random.PRNGKey(seed)
        else:
            self._rng = np.random.default_rng(seed)
        self.cell_discretization = cell_discretization

    def _sample_weights_jax(
        self,
        mean_bounds: tuple[float, float],
        variance_bounds: tuple[float, float],
    ) -> jnp.ndarray:
        """
        Sample GMM, evaluate PDF on grid, and normalize weights (JAX).
        """
        pdf, self._key = get_gmm_pdf_jax(
            key=self._key,
            dim=self._dim,
            num_components=self._num_components,
            mean_bounds=mean_bounds,
            variance_bounds=variance_bounds,
        )
        w = pdf(self._points)
        return w / jnp.sum(w)

    def _sample_weights_np(
        self,
        mean_bounds: tuple[float, float],
        variance_bounds: tuple[float, float],
    ) -> np.ndarray:
        """
        Sample GMM parameters using Wishart-based covariances, evaluate PDF, and normalize weights (NumPy/SciPy).
        """
        means_arr, covs_arr, weights = sample_gmm_params_wishart(
            K=self._num_components,
            d=self._dim,
            mean_bounds=mean_bounds,
            wishart_df=self._wishart_df,
            wishart_scale=self._wishart_scale,
            rng=self._rng,
        )
        pdf = build_gmm_pdf_scipy(means_arr, covs_arr, weights)
        w = pdf(np.asarray(self._points))
        return w / np.sum(w)

    def generate(self) -> Iterator[TwoMarginalProblem]:
        axes = get_axes(self._dim, self._borders,
                        self._n_points,
                        cell_discretization=self.cell_discretization,
                        use_jax=self._use_jax)
        self._points = generate_nd_grid(axes, use_jax=self._use_jax)

        mean_bounds = (
            self._borders[0] * MEAN_FROM_BORDERS_COEF,
            self._borders[1] * MEAN_FROM_BORDERS_COEF
        )
        variance_bounds = (
            abs(self._borders[1]) * VARIANCE_LOWER_BOUND_COEF,
            abs(self._borders[1]) * VARIANCE_UPPER_BOUND_COEF
        )

        sampler = (
            self._sample_weights_jax
            if self._use_jax
            else self._sample_weights_np
        )

        for _ in range(self._num_datasets):
            w_mu = sampler(mean_bounds, variance_bounds)
            w_nu = sampler(mean_bounds, variance_bounds)

            # mu = DiscreteMeasure(points=self._points, weights=w_mu)
            # nu = DiscreteMeasure(points=self._points, weights=w_nu)
            mu = _build_measure(self._points, w_mu, axes, self._measure_mode, self._use_jax)
            nu = _build_measure(self._points, w_nu, axes, self._measure_mode, self._use_jax)

            yield TwoMarginalProblem(
                name=self._name,
                mu=mu,
                nu=nu,
                cost_fn=self._cost_fn,
            )
