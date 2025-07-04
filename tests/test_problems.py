import numpy as np
import pytest

from uot.data.measure import DiscreteMeasure
from uot.problems.two_marginal import TwoMarginalProblem
from uot.problems.multi_marginal import MultiMarginalProblem
from uot.problems.base_problem import MarginalProblem
from uot.problems.generators.gaussian_mixture_generator import GaussianMixtureGenerator


def test_two_marginal_problem_basic():
    # Create two tiny discrete measures in 1D
    X = np.array([[0.0], [1.0]])
    a = np.array([0.5, 0.5])
    Y = np.array([[0.0], [2.0]])
    b = np.array([0.4, 0.6])

    mu = DiscreteMeasure(X, a, name="mu")
    nu = DiscreteMeasure(Y, b, name="nu")

    # Define a simple squared‐distance cost function
    def cost_fn(A, B): return np.linalg.norm(
        A[:, None, :] - B[None, :, :], axis=2) ** 2

    prob = TwoMarginalProblem("test2", mu, nu, cost_fn)

    # get_marginals should return the same measures
    marg = prob.get_marginals()
    assert isinstance(marg, list) and len(marg) == 2
    assert np.allclose(marg[0].points, X)
    assert np.allclose(marg[0].weights, a)
    assert np.allclose(marg[1].points, Y)
    assert np.allclose(marg[1].weights, b)

    # get_costs should compute a 2×2 cost matrix:
    costs = prob.get_costs()
    assert isinstance(costs, list) and len(costs) == 1
    C = costs[0]
    assert C.shape == (2, 2)
    # (0–0)^2, (0–2)^2, (1–0)^2, (1–2)^2
    expected_C = np.array([[0.0, 4.0], [1.0, 1.0]])
    assert np.allclose(C, expected_C)

    # Calling get_costs again should return the cached matrix, not recompute
    C2 = prob.get_costs()[0]
    assert C2 is C

    # to_dict should include dataset name and sizes
    d = prob.to_dict()
    assert d["dataset"] == "test2"
    assert d["type"] == "two_marginal"
    assert d["n_mu"] == 2
    assert d["n_nu"] == 2

    # free_memory should clear the cached cost
    prob.free_memory()
    assert prob._C is None
    # Next get_costs should recompute (new object)
    C3 = prob.get_costs()[0]
    assert C3 is not C


@pytest.mark.skip()
def test_multi_marginal_problem_basic():
    # Create three discrete measures in 1D
    X1 = np.array([[0.0], [1.0]])
    w1 = np.array([0.5, 0.5])
    X2 = np.array([[2.0], [3.0]])
    w2 = np.array([0.6, 0.4])
    X3 = np.array([[4.0], [5.0]])
    w3 = np.array([0.7, 0.3])

    m1 = DiscreteMeasure(X1, w1, name="m1")
    m2 = DiscreteMeasure(X2, w2, name="m2")
    m3 = DiscreteMeasure(X3, w3, name="m3")

    # Define a simple 3‐way cost: sum of pairwise squared distances
    def cost_fn_3(A, B, C):
        # A, B, C are (2×1) arrays
        # Build cost tensor of shape (2,2,2)
        n1, n2, n3 = A.shape[0], B.shape[0], C.shape[0]
        T = np.zeros((n1, n2, n3))
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    T[i, j, k] = (
                        (A[i, 0] - B[j, 0]) ** 2
                        + (A[i, 0] - C[k, 0]) ** 2
                        + (B[j, 0] - C[k, 0]) ** 2
                    )
        return T

    prob = MultiMarginalProblem("test3", [m1, m2, m3], [cost_fn_3])

    # get_marginals should return the same three measures
    marg = prob.get_marginals()
    assert isinstance(marg, list) and len(marg) == 3
    assert all(isinstance(mi, DiscreteMeasure) for mi in marg)

    # get_costs should compute a 2×2×2 tensor
    costs = prob.get_costs()
    assert isinstance(costs, list) and len(costs) == 1
    T = costs[0]
    assert T.shape == (2, 2, 2)
    # Check a known entry: i=j=k=0 → all points are [0], so cost = 0
    assert np.isclose(T[0, 0, 0], 0.0)
    # Check i=0,j=0,k=1: cost = (0−2)^2+(0−5)^2+(2−5)^2 = 4 + 25 + 9 = 38
    assert np.isclose(T[0, 0, 1], 38.0)

    # Cached behavior
    T2 = prob.get_costs()[0]
    assert T2 is T

    # to_dict should include type and sizes
    d = prob.to_dict()
    assert d["dataset"] == "test3"
    assert d["type"] == "multi_marginal"
    assert d["sizes"] == [2, 2, 2]

    # free_memory should clear the cached tensor
    prob.free_memory()
    assert prob._C_list == [None]


def test_marginal_problem_is_abstract():
    # Ensure you cannot instantiate MarginalProblem directly
    with pytest.raises(TypeError):
        MarginalProblem("abstract", [], [])


def test_gaussian_mixture_generator_output():
    gen = GaussianMixtureGenerator()
    # Generate 5 two-marginal problems of 3 points in 2D
    problems = gen.generate(n_points=3, dim=2, num_datasets=5)
    assert isinstance(problems, list)
    assert len(problems) == 5
    for p in problems:
        assert isinstance(p, TwoMarginalProblem)
        # Check they have correct names and marginals
        assert p.name.startswith("gauss_")
        marg = p.get_marginals()
        assert len(marg) == 2
        # Each measure has 3 points
        assert marg[0].points.shape[0] == 3
        assert marg[1].points.shape[0] == 3
        # Cost matrix is 3×3
        C = p.get_costs()[0]
        assert C.shape == (3, 3)
