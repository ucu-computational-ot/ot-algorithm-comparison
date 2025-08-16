from collections.abc import Sequence
import jax
import jax.numpy as jnp

from uot.data.measure import DiscreteMeasure
from uot.utils.types import ArrayLike
from uot.solvers.base_solver import BaseSolver

from .method import backnforth_sqeuclidean_nd
from .pushforward import _central_gradient_nd


class BackNForthSqEuclideanSolver(BaseSolver):
    def __init__(self):
        return super().__init__()

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        maxiter: int = 1000,
        tol: float = 1e-6,
        stepsize: float = 1,
    ) -> dict:
        if len(marginals) != 2:
            raise ValueError("Back-and-Forth solver accepts only two marginals.")

        mu, nu = marginals[0], marginals[1]
        coords = [
            mu.to_discrete()[0],
            nu.to_discrete()[0],
        ]
        mu_weights = mu.to_discrete()[1]
        nu_weights = nu.to_discrete()[1]

        iters, phi, psi, rho_nu, rho_mu, errs, duals = backnforth_sqeuclidean_nd(
            mu=mu_weights,
            nu=nu_weights,
            coords=coords,
            stepsize=stepsize,
            maxiterations=maxiter,
            tolerance=tol,
        )

        monge_map = _central_gradient_nd(psi)
        cost = jnp.sum(jnp.sum((coords[0] - monge_map)**2, axis=1) * mu_weights)

        return {
            "monge_map": monge_map,
            "cost": cost,
            "u_final": phi,
            "v_final": psi,
            "iterations": iters,
            "error": errs[iters],
        }
