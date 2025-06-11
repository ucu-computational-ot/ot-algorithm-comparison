import os
import time
import yaml
import argparse

from uot.problems.iterator import ProblemIterator
from uot.problems.store import ProblemStore
from uot.experiments.experiment import Experiment
from uot.experiments.runner import run_pipeline
from uot.utils.import_helpers import import_object
from uot.experiments.solver_config import SolverConfig


def solve_fn(prob, solver, marginals, costs, **kwargs):
    start_time = time.perf_counter()
    metrics = solver().solve(marginals=marginals, costs=costs, **kwargs)
    metrics['transport_plan'].block_until_ready()
    metrics["time"] = (time.perf_counter() - start_time) * 1000
    return metrics


def load_solvers(config: dict) -> list[SolverConfig]:

    solvers_configs = config['solvers']
    params = config['param-grids']

    loaded_solvers_configs = []

    for solver_name, solver_config in solvers_configs.items():
        solver_class = solver_config['solver']
        params_grid_name = solver_config['param-grid']
        is_jit = solver_config['jit']

        solver = import_object(solver_class)

        solver_config = SolverConfig(
            name=solver_name,
            solver=solver,
            param_grid=params[params_grid_name],
            is_jit=is_jit
        ) 

        loaded_solvers_configs.append(solver_config)

    return loaded_solvers_configs


def load_problems(config: dict) -> list[ProblemIterator]:

    iterators = []
    problemsets_names = config['problems']['names']
    problemsets_dir = config['problems']['dir']

    for problemset_name in problemsets_names:
        store_path = os.path.join(problemsets_dir, problemset_name)
        problems_store = ProblemStore(store_path)
        problems_iterator = ProblemIterator(problems_store)
        iterators.append(problems_iterator)
    
    return iterators


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run time precision experiments.")

    parser.add_argument(
        "--save",
        help="Flag to save the results to a CSV file."
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help="Number of folds for cross-validation or repeated experiments."
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to a configuration file with experiment parameters."
    )

    parser.add_argument(
        "--export",
        type=str,
        default="gaussian_toy_results.csv",
        help="Path to export the results CSV file."
    )

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file) 

    experiment = Experiment("Measure time", solve_fn)
    solver_configs = load_solvers(config=config)
    problems_iterators = load_problems(config=config)

    df = run_pipeline(
        experiment=experiment,
        solvers=solver_configs,
        iterators=problems_iterators,
        folds=args.folds,
        progress=True
    )

    print(f"Exporting results to {args.export}")
    df.to_csv(args.export, index=False)
