from jax import numpy as jnp

from ott.geometry import geometry
from ott.solvers import linear

from collections.abc import Sequence

from uot.solvers.base_solver import BaseSolver
from uot.data.measure import DiscreteMeasure


class OTTSinkhornSolver(BaseSolver):
    """
    Sinkhorn solver using ott-jax
    """

    def __init__(
        self,
        epsilon: float = 1e-3,
        max_iterations: int = 1000,
        threshold: float = 1e-6,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.threshold = threshold

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[jnp.ndarray],
        reg: float = None,
        maxiter: int = None,
        tol: float = None,
    ):
        # unpack marginals
        mu = jnp.array(marginals[0].to_discrete()[1])  # shape (n,)
        nu = jnp.array(marginals[1].to_discrete()[1])  # shape (m,)
        C = jnp.array(costs[0])
        # NOTE: with the normalization sinkhorn performs MUCH faster
        C = C / C.max()

        # override defaults if provided
        epsilon = reg if reg is not None else self.epsilon
        max_it = int(maxiter) if maxiter is not None else self.max_iterations
        thr = tol if tol is not None else self.threshold

        # build ott geometry
        geom = geometry.Geometry(cost_matrix=C, epsilon=epsilon)

        # run sinkhorn
        out = linear.solve(
            geom,
            a=mu,
            b=nu,
            threshold=thr,
            max_iterations=max_it,
            lse_mode=True,
        )

        # extract outputs
        coupling = out.matrix              # transport plan
        cost_val = out.reg_ot_cost        # regularized OT cost
        f = out.f                          # dual potential on first measure
        g = out.g                          # dual potential on second measure
        iters = out.n_iters        # number of iterations

        error = jnp.maximum(
            jnp.linalg.norm(coupling.sum(axis=1) - mu),
            jnp.linalg.norm(coupling.sum(axis=0) - nu),
        )

        return {
            "transport_plan": coupling,
            "cost": jnp.sum(coupling * costs[0]),
            "error": error,
            "u_final": f,
            "v_final": g,
            "iterations": iters,
        }
