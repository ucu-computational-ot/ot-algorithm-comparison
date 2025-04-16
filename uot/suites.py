import jax
import time
import numpy as np
from algorithms.sinkhorn import sinkhorn
from memory_profiler import memory_usage, profile
from uot.experiment import OTProblem, Experiment, ExperimentSuite


def precision_experiment(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.to_jax_arrays()
    exact_T, exact_dist = ot_problem.exact_map, ot_problem.exact_cost

    jax.config.update("jax_enable_x64", True)
    
    output_T, output_dist = solver(a, b, C)
    precision = np.abs(output_dist - exact_dist) / exact_dist
    coupling_precision = np.sum(np.abs(output_T - exact_T)) / np.prod(output_T.shape)

    return {'cost_rerr': precision, 'coupling_avg_err': np.mean(coupling_precision.item())}


def time_experiment(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.to_jax_arrays()

    start_time = time.perf_counter()
    _, _ = solver(a, b, C) 
    end_time = time.perf_counter()

    return {'time': (end_time - start_time) * 1000}


# Requires if __name__ == '__main__' to work properly
def memory_experiment(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.to_jax_arrays()

    mem_usage = memory_usage((solver, (a, b, C)), interval=1e-4)

    return {'memory': max(mem_usage) - min(mem_usage)}


def memory_profiler_out(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.to_jax_arrays()

    profiled_solver = profile(solver)
    profiled_solver(a, b, C)


standard_suite = ExperimentSuite(experiments=[
    Experiment("Measure time", run_function=time_experiment),
    Experiment("Measure precision", run_function=precision_experiment),
    Experiment("Measure memory", run_function=memory_experiment)
])

