import time

def measure_time(prob, solver, marginals, costs, **kwargs):
    start_time = time.perf_counter()
    metrics = solver().solve(marginals=marginals, costs=costs, **kwargs)
    metrics["time"] = (time.perf_counter() - start_time) * 1000
    return metrics

