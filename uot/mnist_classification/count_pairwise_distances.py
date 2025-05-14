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

from uot.algorithms.sinkhorn import jax_sinkhorn, ott_jax_sinkhorn
from uot.algorithms.gradient_ascent import gradient_ascent
from uot.algorithms.lbfgs import lbfgs_ot
from uot.algorithms.lp import pot_lp

import multiprocessing as mp
from functools import partial
import argparse
import inspect


solvers = {
    'sinkhorn': jax_sinkhorn,
    'grad-ascent': gradient_ascent,
    'dual-lbfs': lbfgs_ot,
    'ott-sinkhorn': ott_jax_sinkhorn,
    'lp': pot_lp
}

basic_params = {
    'epsilon': 1e-1,
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
            import jax.numpy as jnp
            a = jnp.array(a)
            b = jnp.array(b)
            C = jnp.array(C)
        except Exception as e:
            print(f"Failed to convert to JAX array: {e}")
    
    dist = solver_fn(a, b, C)[1]
    return (i, j, dist)

def compute_distances_parallel(X, C, solver_name, solver_fn, num_processes, epsilon=None):
    n = X.shape[0]
    args = [(i, j) for i in range(n) for j in range(i, n) if i < j]
    
    with mp.Pool(num_processes) as pool:
        worker = partial(compute_distance_pair, X=X, C=C, solver_fn=solver_fn)
        results = list(tqdm(pool.imap_unordered(worker, args),
                            total=len(args),
                            desc=f"Computing distances with {solver_name}"))

    dist_matrix = np.zeros((n, n))
    for i, j, dist in results:
        dist_matrix[i, j] = dist_matrix[j, i] = dist

    epsilon_str = f"_eps{epsilon}" if epsilon is not None else ""
    np.savetxt(f"classification/{solver_name}_{epsilon_str}_pairwise_distances.csv", dist_matrix, delimiter=",")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute pairwise distances using specified solvers.")
    parser.add_argument("--solvers", type=str, nargs='+', choices=solvers.keys(), required=True, 
                        help="Solver name(s) to use. Multiple solvers can be specified.")
    parser.add_argument("--num-processes", type=int, default=4, 
                        help="Number of processes to use in the pool.")
    parser.add_argument("--epsilon", type=float, default=basic_params['epsilon'],
                        help="Regularization parameter for regularized solvers.")
    parser.add_argument("--max-iter", type=int, default=basic_params['max_iter'],
                        help="Maximum number of iterations for iterative solvers.")
    args = parser.parse_args()

    digits = load_digits()
    X, _ = digits.data, digits.target
    X = X / X.sum(axis=1).reshape(X.shape[0],1) + 1e-12
    X = X[:100]

    row, col = np.arange(8), np.arange(8)
    row, col = np.meshgrid(row, col)
    points = np.vstack([coordinate.ravel() for coordinate in [row, col]]).T
    C = ot.dist(points, points).astype('float64')
    C /= C.max()


    solver_params = {
        'epsilon': args.epsilon,
        'max_iter': args.max_iter
    }
    
    num_processes = args.num_processes

    for solver_name in args.solvers:
        solver_fn = get_solver_with_params(solvers[solver_name], **solver_params)
        compute_distances_parallel(X, C, solver_name, solver_fn, num_processes, solver_params['epsilon'])