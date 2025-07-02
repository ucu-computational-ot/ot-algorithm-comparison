from uot.experiments.experiment import Experiment
from uot.experiments.runner import run_pipeline
from uot.solvers.solver_config import SolverConfig
from uot.problems.generators.gaussian_mixture_generator import GaussianMixtureGenerator
from uot.solvers.sinkhorn import SinkhornTwoMarginalSolver

from uot.utils.costs import cost_euclid_squared

import time


def solve_fn(prob, solver, marginals, costs, **kwargs):
    start_time = time.perf_counter()
    metrics = solver.solve(marginals, costs, **kwargs)
    metrics["time"] = time.perf_counter() - start_time
    return metrics


if __name__ == "__main__":
    gauss_gen1 = GaussianMixtureGenerator(
        name="Gaussians (100 pts)",
        dim=1,
        num_components=1,
        n_points=100,
        num_datasets=5,
        borders=(-6, 6),
        cost_fn=cost_euclid_squared,
        use_jax=True,
        seed=43,
    )
    gauss_gen2 = GaussianMixtureGenerator(
        name="Gaussians (1000pts)",
        dim=1,
        num_components=1,
        n_points=1000,
        num_datasets=5,
        borders=(-6, 6),
        cost_fn=cost_euclid_squared,
        use_jax=True,
        seed=43,
    )

    experiment = Experiment("GaussianToy", solve_fn)
    sinkhorn_config = SolverConfig(
        name="Sinkhorn",
        solver=SinkhornTwoMarginalSolver(),
        param_grid=[
            {"reg": 1e-2},
            {"reg": 1e-3},
        ],
        is_jit=True,
    )
    df = run_pipeline(
        experiment=experiment,
        solvers=[sinkhorn_config],
        iterators=[gauss_gen1, gauss_gen2],
        folds=10,
    )

    filename = "gaussian_toy_results.csv"
    print(f"Exporting results to {filename}")
    df.to_csv(filename, index=False)
