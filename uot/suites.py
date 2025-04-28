import jax
import jaxlib
import time
import jaxlib.xla_extension
import numpy as np
from algorithms.sinkhorn import sinkhorn
from memory_profiler import memory_usage, profile
from uot.experiment import OTProblem, Experiment, ExperimentSuite

def time_precision_experiment(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.to_jax_arrays()
    start_time = time.perf_counter()
    output_T, output_dist = solver(a, b, C)
    end_time = time.perf_counter()

    exact_T, exact_dist = ot_problem.exact_map, ot_problem.exact_cost
    precision = np.abs(output_dist - exact_dist) / exact_dist
    coupling_precision = np.sum(np.abs(output_T - exact_T)) / np.prod(output_T.shape)

    return {'time': (end_time - start_time) * 1000, 'cost_rerr': precision, 'coupling_avg_err': coupling_precision}


# Requires if __name__ == '__main__' to work properly
def memory_experiment(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.to_jax_arrays()

    mem_usage = memory_usage((solver, (a, b, C)), interval=1e-3)

    return {'memory': max(mem_usage)}


def memory_profiler_out(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.to_jax_arrays()

    profiled_solver = profile(solver)
    profiled_solver(a, b, C)


time_precision_suite = ExperimentSuite(experiments=[
    Experiment("Measure time and precision", run_function=time_precision_experiment),
    # Experiment("Measure time and precision", run_function=time_precision_experiment),
])

memory_suite = ExperimentSuite(experiments=[
    Experiment("Measure memory", run_function=memory_experiment)
])

