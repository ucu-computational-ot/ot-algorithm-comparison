import numpy as np
from uot.utils.types import ArrayLike
from uot.problems.problem_generator import ProblemGenerator
from uot.utils.generate_nd_grid import generate_nd_grid, compute_cell_volume
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


# MEAN_FROM_BORDERS_COEF = 0.5
MEAN_FROM_BORDERS_COEF = 0.2
VARIANCE_LOWER_BOUND_COEF = 0.001
# VARIANCE_UPPER_BOUND_COEF = 0.5
VARIANCE_UPPER_BOUND_COEF = 0.01


class GaussianMixtureGenerator(ProblemGenerator):
    """
    Generate two-marginal optimal transport problems whose marginals are sampled
    from Gaussian mixture models (GMMs) and discretized on a fixed Cartesian grid.

    Each yielded dataset item is a :class:`uot.problems.two_marginal.TwoMarginalProblem`
    with two discrete probability measures (mu, nu) supported on the same grid points.

    Overview
    --------
    Let the domain be a hyper-rectangle [b0, b1]^d and let {x_i}_{i=1}^N be a uniform
    grid (N = n_points^d). For each marginal, the generator samples a GMM density

        p(x) = sum_{k=1}^K alpha_k * N(x; mu_k, Sigma_k),

    evaluates p on the grid, and converts these values into discrete weights w_i.
    If using cell-centered discretization, the generator multiplies by the grid
    cell volume ΔV to approximate the continuous mass integral via a Riemann sum:

        w_i ∝ p(x_i) * ΔV   (cell-centered)
        w_i ∝ p(x_i)        (vertex-centered or density-as-discrete)

    and then normalizes so that sum_i w_i = 1.

    Sampling backends
    -----------------
    Two sampling/evaluation backends are supported:

    * JAX path (use_jax=True):
      Uses `get_gmm_pdf` to sample parameters and return a JAX-callable pdf.
      Randomness is driven by a JAX PRNGKey.

    * NumPy/SciPy path (use_jax=False):
      Samples GMM parameters using Wishart-based random covariances and builds a
      SciPy pdf. Covariances are re-scaled so each component has a controlled
      average variance (trace/d) within `variance_bounds`.

    Parameters
    ----------
    name:
        Name assigned to each generated TwoMarginalProblem.
    dim:
        Ambient dimension (must be 1, 2, or 3).
    num_components:
        Number of mixture components K in each GMM.
    n_points:
        Number of grid points per axis (total support size is n_points**dim).
    num_datasets:
        Number of problem instances to generate (iterator length).
    borders:
        Tuple (b0, b1) defining the (shared) domain bounds on each axis.
    cost_fn:
        Callable cost function c(x, y) used by TwoMarginalProblem.
    use_jax:
        If True, use JAX sampling/pdf evaluation; otherwise use NumPy/SciPy.
    seed:
        Random seed for the backend RNG (JAX PRNGKey or NumPy Generator).
    wishart_df:
        Degrees of freedom for Wishart sampling in the NumPy path. Defaults to dim+1.
    wishart_scale:
        Scale matrix for Wishart sampling in the NumPy path. Defaults to I_d.
    mean_from_borders_coef:
        Fraction of the domain width used as a margin to keep component means away
        from borders. Means are sampled within:
            (b0 + (b1-b0)*coef, b1 - (b1-b0)*coef).
    variance_lower_bound_coef, variance_upper_bound_coef:
        Coefficients used to derive variance bounds relative to |b1|:
            variance_bounds = (|b1|*lower_coef, |b1|*upper_coef).
        These bounds control the typical component scale via covariance re-scaling.
    measure_mode:
        Measure construction mode passed to `_build_measure`:
        typically 'grid', 'discrete', or 'auto' (project-specific semantics).
    cell_discretization:
        Grid discretization convention: 'cell-centered' or 'vertex-centered'.
        For 'cell-centered', weights are multiplied by cell volume to approximate
        continuous integrals.

    Yields
    ------
    TwoMarginalProblem
        An instance with fields (name, mu, nu, cost_fn), where mu and nu are
        discrete probability measures on the shared grid.

    Notes
    -----
    * The generator draws mu and nu independently at each iteration.
    * The discretization step ensures nonnegative weights and normalizes them to
      a probability simplex (sum to 1).
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
        mean_from_borders_coef: float = MEAN_FROM_BORDERS_COEF,
        variance_lower_bound_coef: float = VARIANCE_LOWER_BOUND_COEF,
        variance_upper_bound_coef: float = VARIANCE_UPPER_BOUND_COEF,
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
        self._mean_from_borders_coef = mean_from_borders_coef
        self._variance_lower_bound_coef = variance_lower_bound_coef
        self._variance_upper_bound_coef = variance_upper_bound_coef
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

    def _rescale_covariances(
        self,
        covs_arr: np.ndarray,
        variance_bounds: tuple[float, float],
    ) -> np.ndarray:
        low, high = variance_bounds
        if low > 0 and high > 0:
            log_low = np.log(low)
            log_high = np.log(high)
            target_vars = np.exp(
                self._rng.uniform(log_low, log_high, size=covs_arr.shape[0])
            )
        else:
            target_vars = self._rng.uniform(low, high, size=covs_arr.shape[0])

        for k in range(covs_arr.shape[0]):
            avg_var = np.trace(covs_arr[k]) / covs_arr.shape[1]
            if avg_var <= 0:
                continue
            covs_arr[k] = covs_arr[k] * (target_vars[k] / avg_var)
        return covs_arr

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
        covs_arr = self._rescale_covariances(covs_arr, variance_bounds)
        pdf = build_gmm_pdf_scipy(means_arr, covs_arr, weights)
        w = pdf(np.asarray(self._points))
        return w / np.sum(w)

    def generate(self) -> Iterator[TwoMarginalProblem]:
        axes = get_axes(
            self._dim,
            self._borders,
            self._n_points,
            cell_discretization=self.cell_discretization,
            use_jax=self._use_jax,
        )
        self._points = generate_nd_grid(axes, use_jax=self._use_jax)
        cell_volume = compute_cell_volume(axes, use_jax=self._use_jax)

        mean_bounds = (
            self._borders[0]
            + (self._borders[1] - self._borders[0]) * self._mean_from_borders_coef,
            self._borders[1]
            - (self._borders[1] - self._borders[0]) * self._mean_from_borders_coef,
        )
        variance_bounds = (
            abs(self._borders[1]) * self._variance_lower_bound_coef,
            abs(self._borders[1]) * self._variance_upper_bound_coef,
        )

        sampler = (
            self._sample_weights_jax
            if self._use_jax
            else self._sample_weights_np
        )

        for _ in range(self._num_datasets):
            w_mu = sampler(mean_bounds, variance_bounds)
            w_nu = sampler(mean_bounds, variance_bounds)

            if self.cell_discretization == "cell-centered":
                w_mu = w_mu * cell_volume
                w_nu = w_nu * cell_volume

            w_mu = w_mu / w_mu.sum()
            w_nu = w_nu / w_nu.sum()

            mu = _build_measure(self._points, w_mu, axes, self._measure_mode, self._use_jax)
            nu = _build_measure(self._points, w_nu, axes, self._measure_mode, self._use_jax)

            yield TwoMarginalProblem(
                name=self._name,
                mu=mu,
                nu=nu,
                cost_fn=self._cost_fn,
            )
