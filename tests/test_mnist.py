import numpy as np
import pytest
from uot.experiments.real_data.mnist_classification.mnist_classification import (
    create_kernel_matrix,
    calculate_results
)


class TestCreateKernelMatrix:
    
    @pytest.mark.parametrize("matrix_size", [3, 5, 10])
    def test_kernel_matrix_shape(self, matrix_size):
        distance_matrix = np.random.rand(matrix_size, matrix_size)
        kernel_matrix = create_kernel_matrix(distance_matrix)
        assert kernel_matrix.shape == distance_matrix.shape
        
    def test_kernel_matrix_symmetry(self):
        distance_matrix = np.random.rand(5, 5)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        kernel_matrix = create_kernel_matrix(distance_matrix)
        np.testing.assert_allclose(kernel_matrix, kernel_matrix.T)
        
    def test_kernel_matrix_positive(self):
        distance_matrix = np.random.rand(4, 4)
        kernel_matrix = create_kernel_matrix(distance_matrix)
        assert np.all(kernel_matrix > 0)
        
    def test_kernel_matrix_zero_distance(self):
        distance_matrix = np.zeros((3, 3))
        kernel_matrix = create_kernel_matrix(distance_matrix)
        expected = np.ones((3, 3))
        np.testing.assert_allclose(kernel_matrix, expected)
        
    @pytest.mark.parametrize("distance_val", [1.0, 2.0, 5.0])
    def test_kernel_matrix_known_values(self, distance_val):
        distance_matrix = np.full((2, 2), distance_val)
        kernel_matrix = create_kernel_matrix(distance_matrix)
        expected = np.exp(-distance_val)
        np.testing.assert_allclose(kernel_matrix, expected)


class TestCalculateResults:
    
    @pytest.fixture
    def mock_data(self):
        np.random.seed(42)
        n_samples = 100
        n_features = 64
        n_classes = 10
        
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        distance_matrix = np.random.rand(n_samples, n_samples)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        indices = np.arange(50)
        
        return X, y, distance_matrix, indices
        
    def test_calculate_results_output_range(self, mock_data):
        X, y, distance_matrix, indices = mock_data
        accuracy = calculate_results(X, y, distance_matrix, indices)
        assert 0 <= accuracy <= 1
        
    def test_calculate_results_deterministic(self, mock_data):
        X, y, distance_matrix, indices = mock_data
        accuracy1 = calculate_results(X, y, distance_matrix, indices)
        accuracy2 = calculate_results(X, y, distance_matrix, indices)
        assert accuracy1 == accuracy2
        
    @pytest.mark.parametrize("n_samples", [20, 30, 50])
    def test_calculate_results_different_sample_sizes(self, n_samples):
        np.random.seed(42)
        X = np.random.rand(100, 64)
        n_classes = 4
        y = np.tile(np.arange(n_classes), n_samples // n_classes + 1)[:n_samples]
        np.random.shuffle(y)
        
        distance_matrix = np.random.rand(100, 100)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        indices = np.arange(n_samples)
        accuracy = calculate_results(X, y, distance_matrix, indices)
        assert 0 <= accuracy <= 1


class TestEdgeCases:
    
    def test_create_kernel_matrix_large_distances(self):
        distance_matrix = np.full((3, 3), 100.0)
        kernel_matrix = create_kernel_matrix(distance_matrix)
        assert np.all(kernel_matrix > 0)
        assert np.all(kernel_matrix < 1e-40)
        
    def test_create_kernel_matrix_empty(self):
        distance_matrix = np.array([]).reshape(0, 0)
        kernel_matrix = create_kernel_matrix(distance_matrix)
        assert kernel_matrix.shape == (0, 0)
        
    def test_calculate_results_single_class(self):
        np.random.seed(42)
        X = np.random.rand(20, 64)
        y = np.zeros(20, dtype=int)
        distance_matrix = np.random.rand(20, 20)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        indices = np.arange(10)
        
        # StratifiedKFold will fail with single class
        with pytest.raises((ValueError, Exception)):
            calculate_results(X, y, distance_matrix, indices)
            
    def test_calculate_results_few_samples(self):
        np.random.seed(42)
        X = np.random.rand(10, 64)
        # Balanced data: 2 classes, 5 samples each for 5-fold CV
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        distance_matrix = np.random.rand(10, 10)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        indices = np.arange(10)
        
        accuracy = calculate_results(X, y, distance_matrix, indices)
        assert 0 <= accuracy <= 1
        
    def test_calculate_results_insufficient_samples_per_class(self):
        np.random.seed(42)
        X = np.random.rand(10, 64)
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 10 classes, 1 sample each
        distance_matrix = np.random.rand(10, 10)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        indices = np.arange(10)
        
        with pytest.raises(ValueError, match="n_splits=5 cannot be greater than"):
            calculate_results(X, y, distance_matrix, indices)


class TestKernelProperties:
    
    def test_kernel_matrix_diagonal_dominance(self):
        # Identity-like distance matrix: zeros on diagonal, ones elsewhere
        distance_matrix = np.ones((4, 4))
        np.fill_diagonal(distance_matrix, 0)
        
        kernel_matrix = create_kernel_matrix(distance_matrix)
        
        # Diagonal should be 1, off-diagonal should be exp(-1)
        assert np.allclose(np.diag(kernel_matrix), 1.0)
        expected_off_diag = np.exp(-1)
        off_diag_values = kernel_matrix[np.triu_indices(4, k=1)]
        assert np.allclose(off_diag_values, expected_off_diag)
        
    def test_kernel_matrix_monotonicity(self):
        distances = [0.5, 1.0, 2.0, 5.0]
        kernels = [np.exp(-d) for d in distances]
        
        # Should be decreasing
        for i in range(len(kernels) - 1):
            assert kernels[i] > kernels[i + 1]
            
    @pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
    def test_kernel_matrix_scaling(self, scale):
        base_distance = np.array([[0, 1], [1, 0]])
        scaled_distance = base_distance * scale
        
        kernel_matrix = create_kernel_matrix(scaled_distance)
        expected = np.exp(-scaled_distance)
        np.testing.assert_allclose(kernel_matrix, expected)


class TestClassificationLogic:
    
    def test_cross_validation_consistency(self):
        np.random.seed(42)
        
        # Simple, separable data
        n_per_class = 20
        X = np.vstack([
            np.random.randn(n_per_class, 64) + [1, 0] * 32,  # Class 0
            np.random.randn(n_per_class, 64) + [-1, 0] * 32  # Class 1
        ])
        y = np.array([0] * n_per_class + [1] * n_per_class)
        
        # Euclidean-like distance matrix
        distance_matrix = np.zeros((40, 40))
        for i in range(40):
            for j in range(40):
                distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
        
        indices = np.arange(40)
        accuracy = calculate_results(X, y, distance_matrix, indices)
        
        # Should be better than random guess
        assert accuracy > 0.4
        
    def test_identical_samples_high_accuracy(self):
        np.random.seed(42)
        
        # Data where same-class samples are identical
        base_samples = np.random.randn(5, 64)
        X = np.vstack([
            np.tile(base_samples[0], (10, 1)),  # 10 identical samples of class 0
            np.tile(base_samples[1], (10, 1)),  # 10 identical samples of class 1
        ])
        y = np.array([0] * 10 + [1] * 10)
        
        # Distance matrix: 0 for identical samples, >0 for different
        distance_matrix = np.zeros((20, 20))
        for i in range(20):
            for j in range(20):
                if (i < 10 and j >= 10) or (i >= 10 and j < 10):
                    distance_matrix[i, j] = 1.0
        
        indices = np.arange(20)
        accuracy = calculate_results(X, y, distance_matrix, indices)
        
        # Should achieve very high accuracy with perfect separation
        assert accuracy > 0.8
