from __future__ import annotations
from collections.abc import Sequence
from typing import Literal, Optional, Dict, Any, Tuple

import jax
from jax import lax
import jax.numpy as jnp

from uot.data.measure import GridMeasure
from uot.utils.types import ArrayLike
from uot.solvers.base_solver import BaseSolver
from uot.utils.central_gradient_nd import _central_gradient_nd

from .method import backnforth_sqeuclidean_nd
from .pushforward import _forward_pushforward_nd
from .monge_map import monge_map_from_psi_nd


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

    def __init__(self):
        return super().__init__()

    def solve(
        self,
        marginals: Sequence[GridMeasure],
        costs: Sequence[ArrayLike],         # kept for BaseSolver signature compatability
        maxiter: int = 1_000,
        tol: float = 1e-6,
        stepsize: float = 1,
        error_metric: ErrorMetric = 'h1_psi',
        stepsize_lower_bound: float = 0.01,
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

        # start = time.perf_counter()
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
        )
        # print(f"Time to solve: {time.perf_counter() - start} seconds")

        d = mu_nd.ndim
        shape = mu_nd.shape
        n_vec = jnp.array(shape, dtype=jnp.float32)
        grids = jnp.meshgrid(*axes_mu, indexing="ij")     # list of d arrays, each (*shape)
        X = jnp.stack(grids, axis=-1)                     # (*shape, d)

        # ----- current monge_map computation (kept as-is) -----
        grad_psi = _central_gradient_nd(-psi)              # (d, *shape)
        if grad_psi.shape[0] == len(axes_mu):
            grad_psi = jnp.moveaxis(grad_psi, 0, -1)      # -> (*shape, d)
        monge_map = monge_map_from_psi_nd(psi=-psi)
        monge_map = jnp.moveaxis(monge_map, 0, -1)

        if monge_map.shape != X.shape:
            raise ValueError(f"Monge map shape {monge_map.shape} != grid shape {X.shape}")

        diff = X - monge_map
        cost = jnp.sum(jnp.sum(diff * diff, axis=-1) * mu_nd)

        # marginal L2 (your existing diagnostics)
        # marginal_L2_mu_to_nu = jnp.linalg.norm((rho_mu - nu_nd).ravel())
        # marginal_L2_nu_to_mu = jnp.linalg.norm((rho_nu - mu_nd).ravel())

        # extra = self._extra_metrics(
        #     mu_nd=mu_nd,
        #     nu_nd=nu_nd,
        #     axes_mu=axes_mu,
        #     X=X,
        #     psi=-psi,
        #     T=monge_map,
        #     # grad_psi_moved=grad_psi,   # (*shape, d)
        #     # rho_mu=rho_mu,
        #     # rho_nu=rho_nu,
        #     # dV=dV,
        # )

        # ----- assemble result -----
        out = {
            "monge_map": monge_map,
            "cost": cost,
            "u_final": phi,
            "v_final": psi,
            "iterations": iters,
            "error": errors[iters - 1],
            "marginal_error_L2": jnp.linalg.norm((rho_mu - nu_nd).ravel()),
            # "marginal_error_mu_to_nu": marginal_L2_mu_to_nu,
            # "marginal_error_nu_to_mu": marginal_L2_nu_to_mu,
        }
        # fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        # axs[0,0].imshow(mu_nd, origin='lower')
        # axs[0,1].imshow(nu_nd, origin='lower')
        # axs[1,0].imshow(rho_nu, origin='lower')
        # axs[1,1].imshow(rho_mu, origin='lower')
        # fig.savefig('bfm-measures-pushforwards.png')
        # plt.close(fig)
        # out.update(extra)
        return out