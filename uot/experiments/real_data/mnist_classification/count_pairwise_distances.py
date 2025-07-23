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
from collections.abc import Callable
import argparse
import yaml
import numpy as np
import jax.numpy as jnp



def compute_distances_np(X: ArrayLike,
                             C: ArrayLike,
                             name: str,
                             solver_fn: Callable,
                             param_kwargs: dict,
                             export_folder: str
                             ):

    n = X.shape[0]
    num_pairs = n * (n - 1) // 2
    args_list = [(i, j) for i in range(n) for j in range(i, n) if i < j]
    supp = np.array([[i, j] for i in range(8) for j in range(8)])

    dist_matrix = np.zeros((n, n))

    progress_bar = tqdm(total=num_pairs, desc=f"Running solver: {name} with parameters: {param_kwargs}")

    for i, j in args_list:

        nu = DiscreteMeasure(weights=X[i], points=supp)
        mu = DiscreteMeasure(weights=X[j], points=supp)

        res = solver_fn([nu, mu], [C], **param_kwargs)

        cost = np.sum(res['transport_plan'] * C)

        dist_matrix[i, j] = dist_matrix[j, i] = float(cost)
        progress_bar.update(1)

    param_str = "_".join([f"{k}_{v}" for k, v in param_kwargs.items()])
    filename = f"{name}_{param_str}.csv"

    os.makedirs(export_folder, exist_ok=True)
    file_path = os.path.join(export_folder, filename)

    np.savetxt(file_path, dist_matrix, delimiter=",")
    progress_bar.close()
    logger.info(f"Saved distance matrix to {file_path}")


def compute_distances_jax(X: jnp.ndarray,
                          C: jnp.ndarray,
                          name: str,
                          solver_fn: callable,
                          param_kwargs: dict,
                          export_folder: str,
                          batch_size: int = 10000):
    n = X.shape[0]
    supp = jnp.array([[i, j] for i in range(8) for j in range(8)])

    pairs = jnp.array([[i, j] for i in range(n) for j in range(i + 1, n)], dtype=jnp.int32)

    def solve_single(w1, w2):
        nu = DiscreteMeasure(weights=w1, points=supp)
        mu = DiscreteMeasure(weights=w2, points=supp)
        res = solver_fn([nu, mu], [C], **param_kwargs)
        return jnp.sum(res['transport_plan'] * C)

    solve_batch = jax.jit(jax.vmap(solve_single))

    num_pairs = pairs.shape[0]
    costs_list = []

    progress_bar = tqdm(total=num_pairs, desc=f"Running solver: {name} with parameters: {param_kwargs}")

    for start_idx in range(0, num_pairs, batch_size):
        end_idx = min(start_idx + batch_size, num_pairs)
        batch_pairs = pairs[start_idx:end_idx]

        X1_batch = X[batch_pairs[:, 0]]
        X2_batch = X[batch_pairs[:, 1]]

        costs_batch = solve_batch(X1_batch, X2_batch)
        costs_list.append(costs_batch)

        progress_bar.update(end_idx - start_idx)

    progress_bar.close()

    costs = jnp.concatenate(costs_list)

    dist_matrix = jnp.zeros((n, n))
    dist_matrix = dist_matrix.at[pairs[:, 0], pairs[:, 1]].set(costs)
    dist_matrix = dist_matrix.at[pairs[:, 1], pairs[:, 0]].set(costs)

    param_str = "_".join([f"{k}_{v}" for k, v in param_kwargs.items()])
    filename = f"{name}_{param_str}.csv"
    os.makedirs(export_folder, exist_ok=True)
    file_path = os.path.join(export_folder, filename)

    np.savetxt(file_path, np.array(dist_matrix), delimiter=",")
    print(f"Saved distance matrix to {file_path}")


def compute_distances_for_all_solvers(X: ArrayLike,
                                      C: ArrayLike,
                                      solvers: list[SolverConfig],
                                      batch_size: int,
                                      export_folder: str
                                      ):

    logger.info("Running the MNIST distance calculation...")

    X_jax = jnp.array(X)
    C_jax = jnp.array(C)

    for solver in solvers:

        params = solver.param_grid

        for param_kwargs in params:

            if not solver.is_jit:

                compute_distances_np(X, C, solver.name, solver.solver().solve, param_kwargs, export_folder)
            
            else:
                compute_distances_jax(X_jax, C_jax, solver.name, solver.solver().solve, param_kwargs, export_folder, batch_size)

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
    
    with open(args.config) as file:
        config = yaml.safe_load(file)

    X, y, C = load_mnist_data()

    solver_configs = load_solvers(config=config)
    batch_size = config.get('batch-size', 10000)

    try:
        export_folder = config['output-dir']
    except KeyError:
        logger.error("Output directory not specified in the configuration file.")
        raise ValueError("Configuration file must contain 'output-dir' key.")

    compute_distances_for_all_solvers(X, C, solver_configs, batch_size=batch_size, export_folder=export_folder)
