MAX_ITER_BY_SOLVER = {
    "gradient":       1_000_000,
    "gradient-log":   1_000_000,
    "gradient-plain": 1_000_000,
    "lbfgs":           100_000,
    "sinkhorn":       1_000_000,
    "sinkhorn-log":   1_000_000,
    # add any other solvers here…
}

TOL_BY_SOLVER = {
    "gradient":       1e-6,
    "gradient-log":   1e-6,
    "gradient-plain": 1e-6,
    "lbfgs":          1e-6,
    "sinkhorn":       1e-6,
    "sinkhorn-log":   1e-6,
    # add any other solvers here…
}
