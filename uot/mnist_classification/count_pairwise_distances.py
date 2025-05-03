import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import ot
import numpy as np
from tqdm import tqdm

from sklearn.datasets import load_digits
from sklearn.preprocessing import normalize

from algorithms.sinkhorn import jax_sinkhorn, pot_sinkhorn
from algorithms.gradient_ascent import gradient_ascent
from algorithms.lbfgs import dual_lbfgs
from algorithms.lp import pot_lp

import multiprocessing as mp
from functools import partial
import argparse
import matplotlib.pyplot as plt

digits = load_digits()
X, _ = digits.data, digits.target
X = X / X.sum(axis=1).reshape(X.shape[0],1) + 1e-12
X = X[:100]


solvers = {
    'sinkhorn': partial(jax_sinkhorn, epsilon=1e-1),
    'grad-ascent': partial(gradient_ascent, epsilon=1e-1),
    'dual-lbfs': dual_lbfgs,
    'pot-sinkhorn': partial(pot_sinkhorn, epsilon=1e-1),
    'lp': pot_lp
}

row, col = np.arange(8), np.arange(8)
row, col = np.meshgrid(row, col)
points = np.vstack([coordinate.ravel() for coordinate in [row, col]]).T
C = ot.dist(points, points).astype('float64')
C /= C.max()

pairwise_distances = {}

def compute_distance_pair(coordinates, X, C, solver_fn):
    i, j = coordinates
    dist = solver_fn(X[i], X[j], C)[1]
    return (i, j, dist)

def compute_distances_parallel(X, C, solver_name, solver_fn, num_processes):
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

    np.savetxt(f"classification/{solver_name}_pairwise_distances.csv", dist_matrix, delimiter=",")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute pairwise distances using a specified solver.")
    parser.add_argument("--solver", type=str, choices=solvers.keys(), required=True, help="Solver name to use.")
    parser.add_argument("--num-processes", type=int, default=4, help="Number of processes to use in the pool.")
    args = parser.parse_args()

    solver_name = args.solver
    solver_fn = solvers[solver_name]
    num_processes = args.num_processes

    compute_distances_parallel(X, C, solver_name, solver_fn, num_processes)