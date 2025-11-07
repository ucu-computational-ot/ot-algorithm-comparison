import jax
from jax import numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
import optax

import pandas as pd

from collections.abc import Sequence
from functools import partial

from uot.problems.generators import GaussianMixtureGenerator
from uot.utils.costs import cost_euclid_squared
from uot.experiments.experiment import Experiment
from uot.experiments.measurement import measure_with_gpu_tracker
from uot.solvers.base_solver import BaseSolver
from uot.data.measure import DiscreteMeasure
from uot.utils.types import ArrayLike
from uot.problems.iterator import OnlineProblemIterator


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def gradient(
    mu: jnp.ndarray,
    nu: jnp.ndarray,
    C: jnp.ndarray,
    reg: float,
    maxiter: int,
    tolerance: float,
    alpha: float,
):
    pass


@partial(jax.jit, static_argnums=(4, 6))
def _sgd(
    mu: jnp.array,
    nu: jnp.array,
    C: jnp.array,
    reg: float,
    maxiter: int,
    tolerance: float,
    optimizer: optax.GradientTransformation,
) -> tuple[
    jnp.array,      # plan
    jnp.array,      # phi
    jnp.array,      # psi
    float,          # cost
    float,          # error
    int,            # iterations
]:
    maxiter = int(maxiter)      # explicitly cast to int for the tracer
    n, m = mu.shape[0], nu.shape[0]
    theta0 = jnp.zeros((n + m,))
    opt_state = optimizer.init(theta0)
    i0, error0 = 0, jnp.inf
    init_state = (i0, theta0, opt_state, error0)

    def cond_fn(state):
        i, _, _, error = state
        return (i < maxiter) & (error > tolerance)

    def step(state):
        i, theta, opt_state, error = state
        phi = theta[:n]
        psi = theta[n:]
        log_K = (phi[:, None] + psi[None, :] - C) / reg
        E1 = jnp.exp(log_K - logsumexp(log_K, axis=1, keepdims=True))
        grad_phi = mu - E1.sum(axis=1)
        E2 = jnp.exp(log_K - logsumexp(log_K, axis=0, keepdims=True))
        grad_psi = nu - E2.sum(axis=0)
        # apply optimizer update
        grad = jnp.concatenate([-grad_phi, -grad_psi])
        updates, opt_state = optimizer.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        def compute_error(_):
            phi = theta[:n]
            psi = theta[n:]
            P = jnp.exp((phi[:, None] + psi[None, :] - C) / reg)
            return jnp.maximum(
                jnp.linalg.norm(P.sum(axis=1) - mu),
                jnp.linalg.norm(P.sum(axis=0) - nu),
            )
        error = lax.cond(
            (i % 10) == 0,
            compute_error,
            lambda _: error,
            operand=None,
        )
        return (i+1), theta, opt_state, error

    iterations, theta, _, error = lax.while_loop(
        cond_fn, step, init_state
    )
    phi = theta[:n]
    psi = theta[n:]
    plan = jnp.exp((phi[:, None] + psi[None, :] - C) / reg)
    cost = jnp.sum(plan * C)
    return plan, phi, psi, cost, error, iterations


class SGDSolver(BaseSolver):
    def __init__(
        self,
        learning_rate: float = 0.003,
        momentum: float = 0.9,
    ):
        super().__init__()
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            decay_rate=0.99,
            transition_steps=1_000,
            staircase=False,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(learning_rate=schedule, momentum=momentum, nesterov=True)
        )

    def solve(
        self,
        marginals: Sequence[DiscreteMeasure],
        costs: Sequence[ArrayLike],
        reg: float = 1e-3,
        maxiter: int = 1000,
        tol: float = 1e-6,
        *args,
        **kwargs,
    ):
        mu, nu = (
            marginals[0].to_discrete()[1],
            marginals[1].to_discrete()[1],
        )
        try:
            plan, phi, psi, cost, error, iterations = _sgd(
                mu=mu,
                nu=nu,
                C=costs[0],
                reg=reg,
                maxiter=maxiter,
                tolerance=tol,
                optimizer=self.optimizer,
            )
        except Exception as e:
            print(e)
        return {
            "transport_plan": plan,
            "cost": cost,
            "u_final": phi,
            "v_final": psi,
            "iterations": iterations,
            "error": error,
        }


def main():

    problems_num = 8

    def get_problem_iterator():
        gen = GaussianMixtureGenerator(
            name="gaussians",
            dim=1,
            num_components=1,
            n_points=32,
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

    params_list = [
        # {
        #     'reg': 0.01,
        #     'maxiter': 1e6,
        #     'tol': 1e-6,
        #     'learning_rate': 0.01,
        # },
        {
            'reg': 0.01,
            'maxiter': 1e7,
            'tol': 1e-6,
            'learning_rate': 0.003,
        },
    ]

    experiment = Experiment(
        name="Test Gradient Implementations",
        solve_fn=measure_with_gpu_tracker,
    )
    results = pd.DataFrame()
    for params in params_list:
        run_results = experiment.run_on_problems(
            problems=get_problem_iterator(),
            solver=SGDSolver,
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
    print()

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
