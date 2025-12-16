from __future__ import annotations
from collections.abc import Sequence, Callable
from typing import Literal, Dict

from jax import lax
import jax.numpy as jnp

from uot.data.measure import GridMeasure
from uot.utils.types import ArrayLike
from uot.solvers.base_solver import BaseSolver
from uot.utils.central_gradient_nd import _central_gradient_nd
from uot.utils.import_helpers import import_object

from uot.utils.metrics.pushforward_map_metrics import extra_grid_metrics

from .method import backnforth_sqeuclidean_nd
from .forward_pushforward import cic_pushforward_nd
from .pushforward import adaptive_pushforward_nd
from .monge_map import (
    monge_map_from_psi_nd,
    monge_map_cic_from_psi_nd,
    monge_map_adaptive_from_psi_nd,
)


ErrorMetric = Literal[
    "tv_psi", "tv_phi", "l_inf_psi",
    "h1_psi", "h1_psi_relative",
    "transportation_cost", "transportation_cost_relative"
]


class BackNForthSqEuclideanSolver(BaseSolver):

    """
    Quadratic-cost Back-and-Forth method (BFM) on tensor grids.
    Marginals must use cell-centered discretization for stability.
    """

    _PUSHFORWARD_ALIASES: Dict[str, Callable] = {
        "adaptive": adaptive_pushforward_nd,
        "adaptive_pushforward_nd": adaptive_pushforward_nd,
        "cic": cic_pushforward_nd,
        "cic_pushforward_nd": cic_pushforward_nd,
        "forward": cic_pushforward_nd,
        "forward_pushforward": cic_pushforward_nd,
        "_forward_pushforward_nd": cic_pushforward_nd,
    }

    def __init__(self,
                 pushforward_fn=adaptive_pushforward_nd,
                 ):
        resolved_fn, resolved_name = self._resolve_pushforward_fn(pushforward_fn)
        self._pushforward_fn = resolved_fn
        self._pushforward_fn_name = resolved_name
        return super().__init__()

    def solve(
        self,
        marginals: Sequence[GridMeasure],
        costs: Sequence[ArrayLike],         # kept for BaseSolver signature compatability
        *args,
        maxiter: int = 1_000,
        tol: float = 1e-6,
        stepsize: float = 1,
        error_metric: ErrorMetric = 'h1_psi',
        stepsize_lower_bound: float = 0.01,
        **kwargs,
    ) -> dict:
        if len(marginals) != 2:
            raise ValueError("Back-and-Forth solver accepts only two marginals.")

        mu, nu = marginals[0], marginals[1]
        axes_mu, mu_nd = mu.for_grid_solver(backend="jax", dtype=jnp.float64)
        axes_nu, nu_nd = nu.for_grid_solver(backend="jax", dtype=jnp.float64)

        # ----- grid helpers -----
        def _grid_spacings(axes):
            # assumes monotone axes, uniform per axis
            hs = [ax[1] - ax[0] if ax.shape[0] > 1 else 1.0 for ax in axes]
            return hs, jnp.prod(jnp.asarray(hs))
        
        hs, dV = _grid_spacings(axes_mu)
        self._hs = hs

        (
            iters,
            phi,
            psi,
            rho_nu,
            rho_mu,
            errors,
            dual_values,
            sigmas,
        ) = backnforth_sqeuclidean_nd(
            mu=mu_nd,
            nu=nu_nd,
            coordinates=axes_mu,
            stepsize=stepsize / jnp.maximum(mu_nd.max(), nu_nd.max()),
            maxiterations=maxiter,
            tolerance=tol,
            progressbar=False,
            stepsize_lower_bound=stepsize_lower_bound,
            error_metric=error_metric,
            pushforward_fn=self._pushforward_fn,
        )

        d = mu_nd.ndim
        shape = mu_nd.shape
        n_vec = jnp.array(shape, dtype=jnp.float32)
        grids = jnp.meshgrid(*axes_mu, indexing="ij")     # list of d arrays, each (*shape)
        X = jnp.stack(grids, axis=-1)                     # (*shape, d)

        # ----- current monge_map computation (kept as-is) -----
        grad_psi = _central_gradient_nd(-psi)              # (d, *shape)
        if grad_psi.shape[0] == len(axes_mu):
            grad_psi = jnp.moveaxis(grad_psi, 0, -1)      # -> (*shape, d)
        monge_map_fn = monge_map_from_psi_nd
        if self._pushforward_fn is adaptive_pushforward_nd:
            monge_map_fn = monge_map_adaptive_from_psi_nd
        elif self._pushforward_fn is cic_pushforward_nd:
            monge_map_fn = monge_map_cic_from_psi_nd
        monge_map = monge_map_fn(psi=-psi)
        monge_map = jnp.moveaxis(monge_map, 0, -1)

        if monge_map.shape != X.shape:
            raise ValueError(f"Monge map shape {monge_map.shape} != grid shape {X.shape}")

        # Convert Monge map from index coordinates back to physical grid coordinates
        spacing_vec = jnp.asarray(hs, dtype=monge_map.dtype)
        origin_vec = jnp.asarray([ax[0] for ax in axes_mu], dtype=monge_map.dtype)
        broadcast_shape = (1,) * d + (d,)
        monge_map_physical = origin_vec.reshape(broadcast_shape) + monge_map * spacing_vec.reshape(broadcast_shape)

        diff = X - monge_map_physical
        cost = jnp.sum(jnp.sum(diff * diff, axis=-1) * mu_nd)


        # ----- assemble result -----
        out = {
            "monge_map": monge_map,
            "cost": cost,
            "u_final": phi,
            "v_final": psi,
            "iterations": iters,
            "error": errors[iters - 1],
            "marginal_error_L2": jnp.linalg.norm((rho_mu - nu_nd).ravel()),
            "pushforward_fn_name": self._pushforward_fn_name,
        }
        return out

    @classmethod
    def _resolve_pushforward_fn(cls, pushforward_fn):
        """
        Accept callables, shorthand aliases or dotted import paths.
        """
        if callable(pushforward_fn):
            return pushforward_fn, getattr(pushforward_fn, "__name__", str(pushforward_fn))

        if isinstance(pushforward_fn, str):
            key = pushforward_fn.lower()
            if key in cls._PUSHFORWARD_ALIASES:
                fn = cls._PUSHFORWARD_ALIASES[key]
                return fn, getattr(fn, "__name__", key)
            fn = import_object(pushforward_fn)
            if callable(fn):
                return fn, getattr(fn, "__name__", pushforward_fn)
            raise TypeError(f"Resolved object '{pushforward_fn}' is not callable.")

        raise TypeError(
            "pushforward_fn must be a callable or an importable string reference, "
            f"got {type(pushforward_fn)}"
        )

    @staticmethod
    def _monge_map_index_to_physical(monge_map, axes):
        spatial_shape = tuple(len(ax) for ax in axes)
        d = len(spatial_shape)
        arr = jnp.asarray(monge_map)
        if arr.ndim == len(spatial_shape):
            arr = arr[..., None]
        if arr.shape[0] == d and arr.ndim == len(spatial_shape) + 1:
            arr = jnp.moveaxis(arr, 0, -1)
        elif arr.shape[-1] != d:
            arr = arr.reshape(spatial_shape + (d,))

        spacings = jnp.array(
            [float(ax[1] - ax[0]) if ax.shape[0] > 1 else 1.0 for ax in axes],
            dtype=arr.dtype,
        )
        origins = jnp.array([float(ax[0]) for ax in axes], dtype=arr.dtype)
        reshape = (1,) * len(spatial_shape) + (d,)
        return origins.reshape(reshape) + arr * spacings.reshape(reshape)

    def _extra_metrics(
        self,
        *,
        mu_nd,
        nu_nd,
        axes_mu,
        X,
        psi,
        T,
    ) -> Dict[str, float]:
        psi_arr = jnp.asarray(psi).reshape(mu_nd.shape)
        pushforward_mu, _ = self._pushforward_fn(mu_nd, psi_arr)
        T_phys = self._monge_map_index_to_physical(T, axes_mu)
        metrics = extra_grid_metrics(
            mu_nd=mu_nd,
            nu_nd=nu_nd,
            axes_mu=axes_mu,
            X=X,
            T=T_phys,
            pushforward_mu=pushforward_mu,
        )
        return {key: float(val) for key, val in metrics.items()}
