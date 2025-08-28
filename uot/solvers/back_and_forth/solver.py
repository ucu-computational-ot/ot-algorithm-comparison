from collections.abc import Sequence
import jax
import jax.numpy as jnp

from uot.data.measure import GridMeasure
from uot.utils.types import ArrayLike
from uot.solvers.base_solver import BaseSolver

from .method import backnforth_sqeuclidean_nd
from .pushforward import _central_gradient_nd


class BackNForthSqEuclideanSolver(BaseSolver):
    def __init__(self):
        return super().__init__()

    def solve(
        self,
        marginals: Sequence[GridMeasure],
        costs: Sequence[ArrayLike],
        maxiter: int = 1000,
        tol: float = 1e-6,
        stepsize: float = 1,
    ) -> dict:
        if len(marginals) != 2:
            raise ValueError("Back-and-Forth solver accepts only two marginals.")

        mu, nu = marginals[0], marginals[1]
        axes_mu, mu_nd = mu.for_grid_solver(backend="jax", dtype=jnp.float64)
        axes_nu, nu_nd = nu.for_grid_solver(backend="jax", dtype=jnp.float64)

        iters, phi, psi, rho_nu, rho_mu, errs, duals = backnforth_sqeuclidean_nd(
            mu=mu_nd,
            nu=nu_nd,
            coordinates=axes_mu,
            stepsize=stepsize,
            maxiterations=maxiter,
            tolerance=tol,
        )

        def _grid_spacings(axes):
            # assumes monotone axes, uniform per axis
            hs = [ax[1] - ax[0] if ax.shape[0] > 1 else 1.0 for ax in axes]
            return hs, jnp.prod(jnp.asarray(hs))

        hs, dV = _grid_spacings(axes_mu)

        grids = jnp.meshgrid(*axes_mu, indexing="ij")                      # list of d arrays, each (*shape)
        X = jnp.stack(grids, axis=-1)                                      # (*shape, d)

        grad_psi = _central_gradient_nd(psi)
        # grad_psi = _central_gradient_nd(phi)
        # print(f"{coords[0].shape=} {monge_map.shape=}")
        # cost = jnp.sum(jnp.sum((coords[0] - monge_map)**2, axis=1) * mu_weights)

        if grad_psi.shape[0] == len(axes_mu):
            grad_psi = jnp.moveaxis(grad_psi, 0, -1)  # (d, n0, n1) → (n0, n1, d)

        monge_map = grad_psi

        if monge_map.shape != X.shape:
            raise ValueError(f"Monge map shape {monge_map.shape} != grid shape {X.shape}")
        diff = X - monge_map                                               # (*shape, d)
        cost = jnp.sum(jnp.sum(diff * diff, axis=-1) * mu_nd)             # scalar

        # Optional: mass-fix after pushforwards
        # mass_mu = jnp.sum(mu_nd) * dV
        # mass_nu = jnp.sum(nu_nd) * dV
        # mass_rho_mu = jnp.sum(rho_mu) * dV
        # mass_rho_nu = jnp.sum(rho_nu) * dV
        # rho_mu = rho_mu * (mass_mu / jnp.maximum(1e-30, mass_rho_mu))
        # rho_nu = rho_nu * (mass_nu / jnp.maximum(1e-30, mass_rho_nu))

        # marginal_error_mu_to_nu = jnp.linalg.norm(rho_mu - nu_nd)
        # marginal_error_nu_to_mu = jnp.linalg.norm(rho_nu - mu_nd)
        marginal_L2_mu_to_nu = jnp.linalg.norm((rho_mu - nu_nd).ravel()) * jnp.sqrt(dV)
        marginal_L2_nu_to_mu = jnp.linalg.norm((rho_nu - mu_nd).ravel()) * jnp.sqrt(dV)

        return {
            "monge_map": monge_map,
            "cost": cost,
            "u_final": phi,
            "v_final": psi,
            "iterations": iters,
            "error": errs[iters-1],
            "marginal_error": jnp.linalg.norm((rho_mu - nu_nd).ravel()),
            "marginal_error_mu_to_nu": marginal_L2_mu_to_nu,
            "marginal_error_nu_to_mu": marginal_L2_nu_to_mu,
        }
