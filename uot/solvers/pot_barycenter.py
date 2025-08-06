import numpy as np
import ot
from uot.data.measure import DiscreteMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike

from jax import numpy as jnp
from typing import Sequence


class POTSinkhornBarycenterSolver(BaseSolver):
    def __init__(self):
        return super().__init__()

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        weights: ArrayLike,
        reg: float = 1e-3,
        maxiter: int = 1000,
        tol: float = 1e-6,
    ) -> dict:
        if len(costs) == 0:
            raise ValueError("Cost tensors not defined.")


        A = jnp.asarray([marg.to_discrete()[1] for marg in marginals]).T

        C = costs[0]

        barycenter, couplings, us, vs, i_final, final_err = barycenter_with_plans(
            A,
            C,
            reg,
            weights = weights,
            tol=tol,
            numItermax=maxiter,
        )

        return {
            "transport_plans": couplings,
            "barycenter": barycenter,
            "us_final": us,
            "vs_final": vs,
            "iterations": i_final,
            "error": final_err,
        }

def barycenter_with_plans(A, M, reg, weights=None, tol=1e-9, numItermax=10_000):
    # 1. barycenter with log tracking
    a_bar, log_bar = ot.bregman.barycenter(
        A, M, reg, weights,
        log=True, stopThr=tol, numItermax=numItermax
    )
    n_iter = log_bar.get("niter", log_bar.get("it", -1))
    # 2. plans & potentials
    plans, pots, marg_err = [], [], 0.0
    for ai in A.T:
        G, log = ot.bregman.sinkhorn(
            a_bar, ai, M, reg, log=True, stopThr=tol
        )
        plans.append(G)
        pots.append((log["u"], log["v"]))

        # update marginal error
        row_err = jnp.linalg.norm(G.sum(axis=1) - a_bar, 1)
        col_err = jnp.linalg.norm(G.sum(axis=0) - ai, 1)
        marg_err = max(marg_err, float(max(row_err, col_err)))

    us = np.array(pots[0])
    vs = np.array(pots[1])

    return a_bar, plans, us, vs, int(n_iter), marg_err