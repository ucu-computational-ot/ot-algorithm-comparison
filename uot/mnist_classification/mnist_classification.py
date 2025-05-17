import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
from glob import glob

from sklearn.datasets import load_digits
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import argparse

np.random.seed(42)

SAMPLE_SIZES = [100, 250, 500]

SOLVERS = {
    'sinkhorn': True,
    'grad-ascent': True,
    'lbfs': True,
    'lp': False
}
EPSILONS = [2.0, 1.0, 0.1]


def load_pairwise_distances(solvers, epsilons = None):
    """
    Load pre-computed pairwise distance matrices from CSV files.
    
    Args:
        solvers: Dictionary of solvers to consider
        epsilons: List of epsilon values to filter by, or None to include all
        normalize_distances: Whether to normalize the distance matrices
        
    Returns:
        A dictionary where keys are solver names and values are dictionaries
        with epsilon as keys and the corresponding distance matrix as values.
    """
    pairwise_distances = {}

    distance_files = glob(os.path.join('classification', '*_pairwise_distances.csv'))
    for file_path in distance_files:
        file_name = os.path.basename(file_path)
        parts = file_name.replace('_pairwise_distances.csv', '').split('_eps_')
        
        solver_name = parts[0]
        epsilon = float(parts[1]) if len(parts) > 1 else None

        if solver_name not in solvers:
            continue
        
        if epsilons and epsilon and epsilon not in epsilons:
            continue
        
        if solver_name not in pairwise_distances:
            pairwise_distances[solver_name] = {}
        
        dist_matrix = np.loadtxt(file_path, delimiter=',')
        pairwise_distances[solver_name][epsilon] = dist_matrix

    return pairwise_distances


def create_kernel_matrix(distance_matrix):
    """Convert distance matrix to a proper kernel matrix using RBF transformation with adaptive width"""

    kernel_matrix = np.exp(-distance_matrix)
    return kernel_matrix


def calculate_results(X, y, distances, sample_size, solver_name, epsilon=None):
    """Calculate classification results using proper cross-validation with precomputed kernel"""
    X_indices = np.random.choice(len(X), size=min(int(sample_size), len(X)), replace=False)
    X_sub = X[X_indices]
    y_sub = y[X_indices]

    full_distance_matrix = distances[solver_name][epsilon]
    sub_distance_matrix = full_distance_matrix[np.ix_(X_indices, X_indices)]


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
    parser.add_argument('--solvers', nargs='+', choices=list(SOLVERS.keys()) + ['all'], default=['all'],
                        help="Solvers to use. Can specify multiple solvers or 'all' (default)")
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=SAMPLE_SIZES,
                        help=f"Sample sizes to test. Default: {SAMPLE_SIZES}")
    parser.add_argument('--epsilons', type=float, nargs='+', default=EPSILONS,
                        help=f"Epsilon values to test for regularized solvers. Default: {EPSILONS}")
    parser.add_argument('--output-file', type=str, default='mnist_classification_results.csv',
                        help="Output CSV file name (will be saved in the classification folder)")
    args = parser.parse_args()
    
    selected_solvers = SOLVERS
    if 'all' not in args.solvers:
        selected_solvers = {solver: SOLVERS.get(solver, False) for solver in args.solvers if solver in SOLVERS}

    export_folder = os.path.join('classification')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder, exist_ok=True)
        print(f"Created output directory: {export_folder}")
        
    digits = load_digits()
    X, y = digits.data, digits.target
    X = normalize(X, axis=1)

    pairwise_distances = load_pairwise_distances(selected_solvers, args.epsilons)
    
    results = []
    
    for sample_size in args.sample_sizes:
        for solver_name in selected_solvers:
            if solver_name not in pairwise_distances:
                print(f"  No distance matrix found for {solver_name}. Skipping.")
                continue
                
            for epsilon in args.epsilons if selected_solvers[solver_name] else [None]:
                if epsilon and epsilon not in pairwise_distances[solver_name]:
                    print(f"  No distance matrix found for {solver_name} with epsilon={epsilon}. Skipping.")
                    continue
                    
                accuracy = calculate_results(X, y, pairwise_distances, sample_size, solver_name, epsilon)
                results.append({
                    'solver': solver_name, 
                    'epsilon': epsilon, 
                    'sample_size': sample_size, 
                    'accuracy': accuracy
                })

    
    results = pd.DataFrame(results)
    output_path = os.path.join(export_folder, args.output_file)
    results.to_csv(output_path, index=False)
