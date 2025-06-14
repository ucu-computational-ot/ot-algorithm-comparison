import os
import jax
jax.config.update("jax_enable_x64", True)

from uot.data.measure import DiscreteMeasure
from uot.utils.mnist_helpers import load_mnist_data
from uot.utils.yaml_helpers import load_solvers
from uot.utils.logging import logger
from uot.utils.types import ArrayLike
from uot.solvers.solver_config import SolverConfig

from tqdm import tqdm
from typing import List, Callable
import argparse
import yaml
import numpy as np
import jax.numpy as jnp



def compute_distances_solver(X: ArrayLike,
                             C: ArrayLike,
                             name: str,
                             solver_fn: Callable,
                             param_kwargs: dict,
                             is_jit: bool,
                             export_folder: str
                             ):

    lib = jnp if is_jit else np

    n = X.shape[0]
    num_pairs = n * (n - 1) // 2
    args_list = [(i, j) for i in range(n) for j in range(i, n) if i < j]
    supp = lib.array([[i, j] for i in range(8) for j in range(8)])

    dist_matrix = np.zeros((n, n))

    progress_bar = tqdm(total=num_pairs, desc=f"Running solver: {name} with parameters: {param_kwargs}")

    if is_jit:
        X = jnp.array(X)
        C = jnp.array(C)

    for i, j in args_list:

        nu = DiscreteMeasure(weights=X[i], points=supp)
        mu = DiscreteMeasure(weights=X[j], points=supp)

        res = solver_fn([nu, mu], [C], **param_kwargs)

        cost = lib.sum(res['transport_plan'] * C)

        dist_matrix[i, j] = dist_matrix[j, i] = float(cost)
        progress_bar.update(1)

    param_str = "_".join([f"{k}_{v}" for k, v in param_kwargs.items()])
    filename = f"{name}_{param_str}.csv"

    os.makedirs(export_folder, exist_ok=True)
    file_path = os.path.join(export_folder, filename)

    np.savetxt(file_path, dist_matrix, delimiter=",")
    progress_bar.close()
    logger.info(f"Saved distance matrix to {file_path}")


def compute_distances_for_all_solvers(X: ArrayLike,
                                      C: ArrayLike,
                                      solvers: List[SolverConfig],
                                      export_folder: str
                                      ):

    logger.info("Running the MNIST distance calculation...")

    for solver in solvers:

        params = solver.param_grid

        for param_kwargs in params:

            compute_distances_solver(X, C , solver.name, solver.solver().solve, param_kwargs, solver.is_jit, export_folder)

    logger.info("All pairwise distances computed successfully.")


if __name__ == "__main__":    

    parser = argparse.ArgumentParser(description="Compute pairwise distances using specified solvers.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a configuration file with experiment parameters."
    )

    args = parser.parse_args()

    X, y, C = load_mnist_data()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file) 

    solver_configs = load_solvers(config=config)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_folder = os.path.join(script_dir, "costs")

    compute_distances_for_all_solvers(X, C, solver_configs, export_folder=export_folder)
