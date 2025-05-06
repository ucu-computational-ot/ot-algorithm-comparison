import random

import numpy as np
from scipy.optimize import linprog
from scipy.linalg import lstsq
from matplotlib import pyplot as plt


##############################################################################################
# Taken from https://github.com/aipyth/gencol_ot
# and modified to work with the current codebase.
##############################################################################################


def place_at_bin(arr, N):
    max_choice = int(N - arr.sum())
    if max_choice <= 0:
        max_choice = 1
    i = random.randint(0, max_choice)
    index = random.choice(np.where(arr == 0)[0])
    arr[index] = i


def generate_random_cols(N, matrix_size):
    A = np.zeros(matrix_size)
    i = 0
    while i < matrix_size[1]:
        while A[:, i].sum() != N:
            place_at_bin(A[:, i], N)
        # check for any repetitions
        for j in range(i):
            if np.all(A[:, j] == A[:, i]):
                A[:, i] = 0
                i -= 1
        i += 1
    return A / N


def initialize_AI(N, grid_size, beta=5):
    A = np.eye(grid_size)
    random_cols = generate_random_cols(N, (grid_size, beta - 1))
    return np.hstack((A, random_cols))


def solve_rmp(AI, cI, marginal):
    result = linprog(cI, A_eq=AI, b_eq=marginal, bounds=(0, None))
    return result.x, result.fun


def solve_rmp_appprox(AI, cI, marginal):
    alpha, residuals, rank, s = lstsq(AI, marginal)
    return alpha


def solve_dual(AI, cI, marginal):
    c_dual = -marginal  # because linprog does minimization
    result = linprog(c_dual, A_ub=AI.T, b_ub=cI,
                     bounds=[(None, 0) for _ in range(AI.shape[0])],
                     method='highs')
    return result.x


def solve_dual_of_rmp(AI, cI, marginal):
    # Negate lambda_star to convert the maximization problem to a minimization problem
    c_dual = -marginal

    # Transpose AI because linprog works with constraints in the form A_ub @ x <= b_ub
    A_ub = AI.T

    # Bounds for y, each y_i should be unrestricted in negative direction and 0 in positive
    # (as `None` indicates no bound and linprog treats variables as non-negative by default)
    bounds = [(None, 0) for _ in range(A_ub.shape[1])]

    # Solve the problem using linprog
    res = linprog(c_dual, A_ub=A_ub, b_ub=cI, bounds=bounds, method='highs')

    # Check if the optimization was successful
    if res.success:
        # Return the optimal value of the original maximization problem by negating res.fun
        optimal_value = -res.fun
        y = res.x
        # print("Optimal dual variables y:", y)
        # print("Maximal value of y^T lambda_star:", optimal_value)
    else:
        print("Optimization failed:", res.message)
        y = None
        optimal_value = None

    return y


def mutate_parent(parent, grid_size):
    child = parent.copy()
    change_index = random.choice(np.where(parent > 0)[0])
    direction = random.choice([-1, 1])
    destination = max(min(grid_size-1, change_index + direction), 0)
    child[destination] += parent[change_index]
    child[change_index] -= parent[change_index]
    child /= child.sum()
    return child


def compute_cost(lambd, N, cost_matrix: np.ndarray):
    return (N**2/2 * lambd.T @ cost_matrix @ lambd -
            N/2 * cost_matrix.diagonal().T @ lambd)


def genetic_column_generation(
        N,
        l,
        beta,
        pair_potential,
        coordinates_of_sites,
        marginal,
        maxiter,
        maxsamples
):
    a, b = coordinates_of_sites
    x = y = np.linspace(a, b, l)
    X, Y = np.meshgrid(x, y, indexing='ij')
    cost_matrix = pair_potential(X, Y)

    # AI = np.eye(l)
    # AI = np.random.randn(l, l)
    # AI /= AI.sum(axis=0)
    AI = initialize_AI(N, l)

    cI = np.empty(AI.shape[1])
    for j in range(AI.shape[1]):
        cI[j] = compute_cost(AI[:, j], N, cost_matrix)
    samples = 0
    iter = 0
    gain = -1

    cost_history = np.empty((maxiter, 1))
    dual_value_history = np.empty((maxiter, l))

    for i in range(maxiter):
        alpha_I, cost = solve_rmp(AI, cI, marginal)
        cost_history[i] = cost
        y_star = solve_dual(AI, cI, marginal)
        dual_value_history[i, :] = y_star

        while gain <= 0 and samples <= maxsamples:
            # Select a random active column of AI
            parent_index = random.choice(np.where(alpha_I > 0)[0])
            parent = AI[:, parent_index]
            child = mutate_parent(parent, l)
            c_child = compute_cost(child, N, cost_matrix)

            # Calculate gain from adding the child column
            gain = np.dot(child.T, y_star) - c_child

            samples += 1

        # Update AI and cI with the new child column if there's a positive gain
        if gain > 0:
            AI = np.hstack((AI, child[:, np.newaxis]))
            # cI = np.hstack((cI, c_child))
            cI = np.append(cI, c_child)
            if AI.shape[1] > beta * l:
                # Clear the oldest inactive columns
                inactive_indices = np.where(alpha_I == 0)[0]
                AI = np.delete(AI, inactive_indices[:l], axis=1)
                cI = np.delete(cI, inactive_indices[:l])

        iter += 1


    # Return the final set of columns and configuration
    alpha_I, cost = solve_rmp(AI, cI, marginal)
    cost_history[i] = cost
    y_star = solve_dual(AI, cI, marginal)
    dual_value_history[i, :] = y_star
    return AI, alpha_I, cost_history, dual_value_history


def col_gen(a, b, C, N=6, maxiter=100, maxsamples=1000, beta=5):
    """
    Column generation algorithm for optimal transport.
    
    Args:
        a (np.ndarray): Source distribution.
        b (np.ndarray): Target distribution.
        C (np.ndarray): Cost matrix.
    
    Returns:
        tuple: (T, cost) where T is the optimal transport matrix and cost is the optimal cost.
    """
    grid_size = len(a)

    
    def pair_potential(x, y, eps=0.1):
        return C
    
    ai, alpha, cost_history, dual_value_history = genetic_column_generation(
        N, grid_size, beta, pair_potential, (0, 1), a, maxiter, maxsamples)
    
    T = np.zeros((len(a), len(b)))
    

    for j in range(ai.shape[1]):
        col_contribution = np.outer(ai[:, j], b) * alpha[j]
        T += col_contribution
    

    T = T / T.sum() * a.sum()
    
    cost = np.sum(T * C)
    
    return T, cost


if __name__ == "__main__":
    N = 6  # Number of marginals
    grid_size = 50  # Number of sites
    beta = 5  # Hyperparameter for controlling the maximum columns
    maxiter = 500  # Maximum number of iterations
    maxsamples = 5000  # Maximum number of samples for mutations
    # maxsamples = 100  # Maximum number of samples for mutations

    grid_points = np.arange(1, grid_size+1)

    marginal = 0.2 + np.power(np.sin(np.pi * grid_points / (grid_size+1)), 2)
    marginal /= marginal.sum()

    def pair_potential(x, y, eps=0.1):
        "Use regulatized Coulomb interaction"
        return 1 / np.sqrt(eps**2 + np.power(x - y, 2))

    print("1-DIMENSIONAL SETTING.")
    print(f"Number of marginals {N=}")
    print(f"Number of grid points {grid_size=}")
    print(f"Hyperparameter for controlling the maximum columns {beta=}")
    print(f"Maximum number of iterations {maxiter=}")
    print(f"First 6 gridpoints {grid_points[:6]}")
    print(f"Last 6 gridpoints {grid_points[-6:]}")
    print(f"Marginal {marginal[:5]}")

    print("=" * 100)

    ai, alpha, cost_history, dual_value_history = genetic_column_generation(
        N, grid_size, beta, pair_potential, (0, 1), marginal, maxiter, maxsamples)

    print(f"{ai.shape=}")
    print(f"{alpha.shape=}")

    gamma = ai @ alpha

    plt.scatter(grid_points, gamma)
    plt.show()