import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import ot
import numpy as np
from tqdm import tqdm

from sklearn.datasets import load_digits

from uot.algorithms.sinkhorn import jax_sinkhorn
from uot.algorithms.gradient_ascent import gradient_ascent
from uot.algorithms.lbfgs import lbfgs_ot
from uot.algorithms.lp import pot_lp

import multiprocessing as mp
from functools import partial
import argparse
import inspect
import jax.numpy as jnp


solvers = {
    'sinkhorn': jax_sinkhorn,
    'grad-ascent': gradient_ascent,
    'lbfs': lbfgs_ot,
    'lp': pot_lp
}

basic_params = {
    'epsilon': [2, 1, 0.1],
    'max_iter': 10000
}



def get_solver_with_params(solver_fn, **kwargs):
    sig = inspect.signature(solver_fn)
    params = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return partial(solver_fn, **params) if params else solver_fn



def compute_distance_pair(coordinates, X, C, solver_fn):
    i, j = coordinates
    a = X[i]
    b = X[j]
    
    fn_name = ""
    if hasattr(solver_fn, "__name__"):
        fn_name = solver_fn.__name__
    elif hasattr(solver_fn, "func") and hasattr(solver_fn.func, "__name__"):
        fn_name = solver_fn.func.__name__
    
    if 'jax' in fn_name:
        try:
            a = jnp.array(a)
            b = jnp.array(b)
            C = jnp.array(C)
        except Exception as e:
            print(f"Failed to convert to JAX array: {e}")
    
    dist = solver_fn(a, b, C)[1]
    return (i, j, dist)



def compute_distances_solver(X, C, solver, num_processes, epsilons, max_iter, progress_bar, export_folder):

    solver_fn = solvers[solver]
    sig = inspect.signature(solver_fn)
    uses_epsilon = 'epsilon' in sig.parameters
    epsilons = epsilons if uses_epsilon else [None]

    for eps in epsilons:

        solver_params = {
            'epsilon': eps,
            'max_iter': max_iter
        } if uses_epsilon else {
            'max_iter': max_iter
        }

        configured_solver = get_solver_with_params(solver_fn, **solver_params)
        n = X.shape[0]
        args_list = [(i, j) for i in range(n) for j in range(i, n) if i < j]
        
        dist_matrix = np.zeros((n, n))
    
        with mp.Pool(num_processes) as pool:

            worker = partial(compute_distance_pair, X=X, C=C, solver_fn=configured_solver)

            for i, j, dist in pool.imap_unordered(worker, args_list):
                dist_matrix[i, j] = dist_matrix[j, i] = dist
                progress_bar.update(1)

        epsilon_str = f"_eps_{eps}"
        np.savetxt(os.path.join(export_folder, f"{solver}{epsilon_str if uses_epsilon else ""}_pairwise_distances.csv"), 
                    dist_matrix, delimiter=",")



def compute_distances_for_all_solvers(X, C, solvers_to_run, num_processes, epsilons,
                                      max_iter, export_folder):
    n = X.shape[0]
    num_pairs = n * (n - 1) // 2
    
    total_jobs = 0
    for solver_name in solvers_to_run:
        solver_fn = solvers[solver_name]
        sig = inspect.signature(solver_fn)
        uses_epsilon = 'epsilon' in sig.parameters
        
        total_jobs += 1 if not uses_epsilon else len(epsilons)
    
    total_distances = num_pairs * total_jobs
    
    progress_bar = tqdm(total=total_distances, desc="Computing all pairwise distances")

    for solver_name in solvers_to_run:
        solver_fn = solvers[solver_name]

        compute_distances_solver(X, C, solver_name, num_processes, epsilons, max_iter, progress_bar, export_folder)
    
    progress_bar.close()

if __name__ == "__main__":    
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="Compute pairwise distances using specified solvers.")
    parser.add_argument("--solvers", type=str, nargs='+', choices=list(solvers.keys()), default=list(solvers.keys()), 
                        help="Solver name(s) to use. Default: uses all solvers. Multiple solvers can be specified.")
    parser.add_argument("--num-processes", type=int, default=4, 
                        help="Number of processes to use in the pool.")
    parser.add_argument("--epsilons", type=float, nargs='+', default=basic_params['epsilon'],
                        help="List of regularization parameters for regularized solvers. Default: [2.0, 1.0, 0.1]")
    parser.add_argument("--max-iter", type=int, default=basic_params['max_iter'],
                        help="Maximum number of iterations for iterative solvers.")
    args = parser.parse_args()

    digits = load_digits()
    X, _ = digits.data, digits.target
    X = X / X.sum(axis=1).reshape(X.shape[0],1) + 1e-12

    row, col = np.arange(8), np.arange(8)
    row, col = np.meshgrid(row, col)
    points = np.vstack([coordinate.ravel() for coordinate in [row, col]]).T
    C = ot.dist(points, points).astype('float64')
    C /= C.max()

    export_folder = "classification"
    if not os.path.exists(export_folder):
        os.makedirs(export_folder, exist_ok=True)
        print(f"Created output directory: {export_folder}")
    
    solver_names_to_run = list(solvers.keys()) if 'all' in args.solvers else args.solvers

    compute_distances_for_all_solvers(X, C, solver_names_to_run, num_processes=args.num_processes,
                                      epsilons=args.epsilons, max_iter=args.max_iter, export_folder=export_folder)