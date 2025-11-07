import jax
from jax import numpy as jnp
from jaxopt import LBFGS

import pandas as pd

from collections.abc import Sequence

from uot.problems.generators import GaussianMixtureGenerator
from uot.utils.costs import cost_euclid_squared
from uot.experiments.experiment import Experiment
from uot.experiments.measurement import measure_with_gpu_tracker
from uot.solvers.base_solver import BaseSolver
from uot.data.measure import DiscreteMeasure
from uot.utils.types import ArrayLike
from uot.problems.iterator import OnlineProblemIterator



@jax.jit
def lbfgs(
    marginals: jnp.ndarray,
    C: jnp.ndarray,
    epsilon: float,
    # core parameters
    tolerance: float,
    maxiter: int,
    history_size: int = 15,
    use_gamma: bool = True,
    # line-search parameters
    maxls: int = 15,            # max inner iterations in LS
    stepsize: float = 0.0       # if \leq 0 runs a line-search, otherwise takes a fixed step
):
    a, b = marginals[0], marginals[1]

    def loss(carry):
        u, v = carry
        K = jnp.exp(
            (u[:, None] + v[None, :] - C) / epsilon
        )
        loss = -jnp.dot(a, u) - jnp.dot(b, v) + epsilon * jnp.sum(K)
        grad = (
            -a + jnp.sum(K, axis=1),
            -b + jnp.sum(K, axis=0),
        )
        return loss, grad

    carry = (jnp.zeros_like(a), jnp.zeros_like(b))
    solver = LBFGS(
        fun=loss,
        maxiter=maxiter,
        tol=tolerance,
        history_size=50,
        value_and_grad=True,
        # implicit_diff=False,
        use_gamma=True,
    )

    solver_result = solver.run(carry)
    u_opt, v_opt = solver_result.params
    P = jnp.exp(
        (u_opt[:, None] + v_opt[None, :] - C) / epsilon
    )
    return u_opt, v_opt, P, solver_result.state.iter_num, solver_result.state.error

class LBFGSSolver(BaseSolver):
    def __init__(self):
        super().__init__()

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        reg: float = 1e-3,
        maxiter: int = 1000,
        tol: float = 1e-6,
        history_size: int = 15,
        use_gamma: bool = True,
        # line-search parameters
        maxls: int = 15,            # max inner iterations in LS
        stepsize: float = 0.0       # if \leq 0 runs a line-search, otherwise takes a fixed step
    ) -> dict:
        mu, nu = (
            marginals[0].to_discrete()[1],
            marginals[1].to_discrete()[1],
        )

        try:
            u, v, transport_plan, iters, grad_norm = lbfgs(
                marginals=jnp.array([mu, nu]),
                C=costs[0],
                epsilon=reg,
                tolerance=tol,
                maxiter=maxiter,
                history_size=history_size,
                use_gamma=use_gamma,
                maxls=maxls,
                stepsize=stepsize,
            )
        except Exception as e:
            print(e)

        marginal_error = jnp.maximum(
            jnp.linalg.norm(transport_plan.sum(axis=1) - mu),
            jnp.linalg.norm(transport_plan.sum(axis=0) - nu),
        )

        return {
            "transport_plan": transport_plan,
            "cost": (transport_plan * costs[0]).sum(),
            "u_final": u,
            "v_final": v,
            "iterations": iters,
            "error": marginal_error,
            "gradient_norm": grad_norm,
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
            'reg': 0.01,
            'maxiter': 1e5,
            'tol': 1e-6,
            'history_size': 20,
            'use_gamma': True,
            'maxls': 15,
        },
        {
            'reg': 0.01,
            'maxiter': 1e5,
            'tol': 1e-6,
            'history_size': 20,
            'use_gamma': True,
            'maxls': 35,
        },
    ]

    experiment = Experiment(
        name="Test LBFGS",
        solve_fn=measure_with_gpu_tracker,
    )
    results = pd.DataFrame()
    for params in lbfgs_params:
        run_results = experiment.run_on_problems(
            problems=get_problem_iterator(),
            solver=LBFGSSolver,
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

    # print(results)
    print()

    grouped_results = results.groupby(['reg', 'maxls'])
    print(
        grouped_results.agg(
            runtime_mean=('time_counter', 'mean'),
            runtime_var=('time_counter', 'var'),
            cost_rerr_min=('cost_rerr', 'min'),
            cost_rerr_max=('cost_rerr', 'max'),
            cost_rerr_mean=('cost_rerr', 'mean'),
        )
    )


if __name__ == "__main__":
    main()
