import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from uot.dataset import generate_1d_gaussians_ds, generate_2d_gaussians_ds, generate_3d_gaussians_ds
from algorithms.sinkhorn import sinkhorn
from uot.experiment import OTProblem, Experiment, get_exact_solution, generate_two_fold_problems 
import time
from memory_profiler import memory_usage, profile


x = np.linspace(-10, 10, 20)
y = np.linspace(-10, 10, 20)
x, y = np.meshgrid(x, y)


gaussians_2d = generate_2d_gaussians_ds(x, y)
ot_problems = generate_two_fold_problems([x, y], gaussians_2d)


def precision_experiment(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.a, ot_problem.b, ot_problem.C
    exact_T, exact_dist = get_exact_solution(a, b, C)


    output_T, output_dist = solver(a, b, C)

    precision = np.abs(output_dist - exact_dist) / exact_dist
    coupling_precision = np.sum(output_T - exact_T) / np.max(np.abs(exact_T))

    return {'precision': precision, 'coupling_precision': np.mean(coupling_precision.item())}


def time_experiment(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.a, ot_problem.b, ot_problem.C

    start_time = time.perf_counter()
    _, _ = solver(a, b, C) 
    end_time = time.perf_counter()

    return {'time': end_time - start_time}


def memory_experiment(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.a, ot_problem.b, ot_problem.C

    mem_usage = memory_usage((solver, (a, b, C)))

    return {'memory': max(mem_usage) - min(mem_usage)}


def memory_profiler_out(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.a, ot_problem.b, ot_problem.C

    profiled_solver = profile(solver)
    profiled_solver(a, b, C)

# def conduct_precisoin_experiment(ot_problem: OTProblem):
#     a, b, C = ot_problem.a, ot_problem.b, ot_problem.C
#     _, exact_dist = get_exact_solution(a, b, C)

#     sinkhorn_coupling = sinkhorn(a, b, C) 
#     sinkhorn_distance = np.sum(C * sinkhorn_coupling)    
#     precision = np.abs(sinkhorn_distance - exact_dist) / exact_dist
#     return {'precision': precision}


# experiment = Experiment("Sinkhorn precision test", ot_problems, run_function=conduct_precisoin_experiment)
# results = experiment.run_experiment()
