import jax
jax.config.update("jax_enable_x64", True)

import ot
import numpy as np
from tqdm import tqdm
from functools import partial

from sklearn.datasets import load_digits
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from algorithms.sinkhorn import jax_sinkhorn
from algorithms.gradient_ascent import gradient_ascent
from algorithms.lbfgs import dual_lbfgs
from algorithms.lp import pot_lp


np.random.seed(32)

SAMPLE_SIZES = [100,
                200,
                300,
                500,
                1000]

digits = load_digits()
X, y = digits.data, digits.target 

X = normalize(X, axis=1)

solvers = {
    'sinkhorn': jax_sinkhorn,
    # 'grad-ascent': gradient_ascent,
    # 'dual-lbfs': dual_lbfgs,
    # 'lp': pot_lp
}

row, col = np.arange(8), np.arange(8)
row, col = np.meshgrid(row, col)
points = np.vstack([coordinate.ravel() for coordinate in [row, col]]).T
C = ot.dist(points, points)

pairwise_distances = {}

with tqdm(total=len(solvers) * len(X) ** 2, desc=f"Computing distances") as pbar:
    for solver_name, solver in solvers.items():
        distances_matrix = np.zeros((X.shape[0], X.shape[0]))
        pairwise_distances[solver_name] = distances_matrix

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i > j:
                    continue
                distance = solver(X[i, :], X[j,:], C)[1]
                distances_matrix[i, j] = distance
                distances_matrix[j, i] = distance
                pbar.update(1)


def ot_kernel(X, Y, solver_name):
    kernel_matrix = np.zeros((X.shape[0], Y.shape[0]))
    distance_matrix = pairwise_distances[solver_name]
    for i, p in enumerate(X):
        for j, q in enumerate(Y):
            pbar.update(1)
            kernel_matrix[i, j] = np.exp(-distance_matrix[i, j])
    return kernel_matrix


for sample_size in SAMPLE_SIZES:
    indices = np.random.choice(X.shape[0], size=int(sample_size), replace=False)

    X_sub = X[indices] 
    y_sub = y[indices]

    kernel = lambda x, y: ot_kernel(x, y, solver_name="jax-sinkhorn")
    pipeline = make_pipeline(SVC(kernel=kernel, C=10, gamma=0.05))
    scores = cross_val_score(pipeline, X_sub, y_sub, cv=5, scoring='accuracy')


y = y.astype(int)
