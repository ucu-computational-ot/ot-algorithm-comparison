from sklearn.datasets import load_digits
import numpy as np
import ot


def load_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the MNIST dataset and return the data, corresponding labels and precomputed cost matrix.
    Returns:
        X (np.ndarray): The MNIST data as a 2D array of shape (n_samples, n_features).
        y (np.ndarray): The labels corresponding to the MNIST data.
        C (np.ndarray): The precomputed cost matrix for the MNIST dataset.
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    X += 1e-6
    X = X / X.sum(axis=1, keepdims=True)

    row, col = np.arange(8), np.arange(8)
    row, col = np.meshgrid(row, col)
    points = np.vstack([coordinate.ravel() for coordinate in [row, col]]).T
    C = ot.dist(points, points).astype('float64')
    C /= C.max()

    return X, y, C
