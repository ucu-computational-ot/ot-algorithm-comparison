from jax import numpy as jnp

from ott.geometry import geometry
from ott.solvers import linear

import pandas as pd

from collections.abc import Sequence

from uot.problems.generators import GaussianMixtureGenerator
from uot.utils.costs import cost_euclid_squared
from uot.experiments.experiment import Experiment
from uot.experiments.measurement import measure_with_gpu_tracker
from uot.solvers.base_solver import BaseSolver
from uot.data.measure import DiscreteMeasure
from uot.problems.iterator import OnlineProblemIterator

import logging
from uot.utils.logging import setup_logger

logger = setup_logger()
logger.setLevel(logging.ERROR)
setup_logger('uot.problems.iterator').setLevel(logging.ERROR)


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
        C = jnp.array(costs[0])                       # shape (n, m)
        C = C / C.sum()

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
            "cost": cost_val,
            "error": error,
            "u_final": f,
            "v_final": g,
            "iterations": iters,
        }


def main():
    problems_num = 31

    def get_problem_iterator():
        gen = GaussianMixtureGenerator(
            name="gaussians",
            dim=1,
            num_components=1,
            n_points=64,
            num_datasets=problems_num,
            borders=(-1, 1),
            cost_fn=cost_euclid_squared,
            use_jax=False,
            seed=55,
        )
        return OnlineProblemIterator(
            generator=gen,
            num=problems_num,
            cache_gt=True,
        )

    lbfgs_params = [
        {
            'reg': 0.0001,
            'maxiter': 1e5,
            'tol': 1e-6,
        },
    ]

    experiment = Experiment(
        name="Test OTT Sinkhorn",
        solve_fn=measure_with_gpu_tracker,
    )
    results = pd.DataFrame()
    for params in lbfgs_params:
        run_results = experiment.run_on_problems(
            problems=get_problem_iterator(),
            solver=OTTSinkhornSolver,
            progress_callback=False,
            **params,
        )
        for key, value in params.items():
            run_results[key] = value
        results = pd.concat([
            results,
            run_results,
        ], ignore_index=True)

    results.drop(index=0, inplace=True)
    results.drop(
        ['time', 'time_unit', 'gpu_mem_unit', 'peak_gpu_mem',
         'combined_peak_gpu_ram', 'peak_gpu_util_pct', 'mean_gpu_util_pct',
         'peak_ram_MiB', 'combined_peak_ram_MiB', 'mean_cpu_util_pct',
         'max_cpu_util_pct'],
        axis=1,
        inplace=True,
    )

    print(results)
    # print()
    #
    # grouped_results = results.groupby(['reg', 'maxls'])
    # print(
    #     grouped_results.agg(
    #         runtime_mean=('time_counter', 'mean'),
    #         runtime_var=('time_counter', 'var'),
    #         cost_rerr_min=('cost_rerr', 'min'),
    #         cost_rerr_max=('cost_rerr', 'max'),
    #         cost_rerr_mean=('cost_rerr', 'mean'),
    #     )
    # )


if __name__ == "__main__":
    main()
