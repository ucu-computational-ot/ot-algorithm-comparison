import os

from uot.utils.mnist_helpers import load_mnist_data
from uot.utils.yaml_helpers import load_solvers
from uot.utils.logging import logger
from uot.utils.types import ArrayLike
from uot.solvers.solver_config import SolverConfig
import numpy as np
import pandas as pd
from datetime import datetime
import yaml

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import argparse

np.random.seed(42)


def get_solver_files(solvers: list[SolverConfig], costs_dir: str)-> list[str]:
    distance_files = []
    for solver in solvers:
        for params in solver.param_grid:

            name = solver.name
            param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
            filename = f"{name}_{param_str}.csv"
            file_path = os.path.join(costs_dir, filename)

            if not os.path.exists(file_path):
                logger.warning(f"Distance file {file_path} does not exist. Skipping.")
                continue
    
            distance_files.append((solver, params, file_path))

    return distance_files



def load_pairwise_distances(solvers: list[SolverConfig], costs_dir: str)-> dict:
    """
    Load pre-computed pairwise distance matrices from CSV files.
    """
    pairwise_distances = {}

    solver_paths = get_solver_files(solvers, costs_dir)
    for solver, params, file_path in solver_paths:
        
        if solver.name not in pairwise_distances:
            pairwise_distances[solver.name] = {}
        
        dist_matrix = np.loadtxt(file_path, delimiter=',')
        pairwise_distances[solver.name][frozenset(params.items())] = dist_matrix

    return pairwise_distances


def create_kernel_matrix(distance_matrix: np.ndarray)-> np.ndarray:
    """Convert distance matrix to a proper kernel matrix"""

    kernel_matrix = np.exp(-distance_matrix)
    return kernel_matrix


def calculate_results(X: ArrayLike, y: ArrayLike, distance: ArrayLike, indices: ArrayLike, solver: SolverConfig)-> float:
    """Calculate classification results using proper cross-validation with precomputed kernel"""
    X_sub = X[indices]
    y_sub = y[indices]

    sub_distance_matrix = distance[np.ix_(indices, indices)]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in cv.split(X_sub, y_sub):
        y_train, y_test = y_sub[train_idx], y_sub[test_idx]

        train_distance = sub_distance_matrix[np.ix_(train_idx, train_idx)]
        test_train_distance = sub_distance_matrix[np.ix_(test_idx, train_idx)]

        train_kernel = create_kernel_matrix(train_distance)
        test_kernel = create_kernel_matrix(test_train_distance)

        clf = SVC(kernel='precomputed', C=10)
        clf.fit(train_kernel, y_train)

        y_pred = clf.predict(test_kernel)
        fold_score = accuracy_score(y_test, y_pred)
        scores.append(fold_score)
    
    return np.mean(scores)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run MNIST classification using precomputed OT distance matrices")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a configuration file with experiment parameters."
    )

    args = parser.parse_args()

    X, y, _ = load_mnist_data()
    X, y = X[:250], y[:250]  # Limit to 250 samples for faster computation

    with open(args.config) as file:
        config = yaml.safe_load(file) 

    solver_configs = load_solvers(config=config)
    costs_dir = config.get('costs-dir', 'results/costs')

    pairwise_distances = load_pairwise_distances(solver_configs, costs_dir)
    
    results = []
    for sample_size in config['sample-sizes']:

        X_indices = np.random.RandomState(42).choice(len(X), size=min(int(sample_size), len(X)), replace=False)

        for solver in solver_configs:
            if solver.name not in pairwise_distances:
                continue

            for param_set, distance_matrix in pairwise_distances[solver.name].items():
                param_kwargs = dict(param_set)

                logger.info(f"Running {solver.name} with parameters {param_kwargs} on sample size {sample_size}")

                accuracy = calculate_results(X, y, distance_matrix, X_indices, solver)

                result = {
                    'solver': solver.name,
                    'sample_size': sample_size,
                    'accuracy': accuracy,
                }

                result.update(param_kwargs)
                results.append(result)

    export_folder = config.get('output-dir', 'results')

    if not os.path.exists(export_folder):
        os.makedirs(export_folder, exist_ok=True)
        logger.info(f"Created output directory: {export_folder}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"mnist_results_{timestamp}.csv"
    
    results = pd.DataFrame(results)
    output_path = os.path.join(export_folder, output_file)
    results.to_csv(output_path, index=False)

    logger.info(f"Results saved to {output_path}")
