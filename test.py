import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from uot.dataset import generate_1d_gaussians_ds, generate_2d_gaussians_ds
from algorithms.sinkhorn import sinkhorn
from uot.experiment import OTProblem, Experiment, get_exact_solution, generate_two_fold_problems 


x = np.linspace(-10, 10, 2)
y = np.linspace(-10, 10, 2)
x, y = np.meshgrid(x, y)
gaussians = generate_2d_gaussians_ds(x, y)
ot_problems = generate_two_fold_problems([x, y], gaussians)

def conduct_precisoin_experiment(ot_problem: OTProblem):
    a, b, c = ot_problem.a, ot_problem.b, ot_problem.C
    _, exact_dist = get_exact_solution(a, b, C)

    sinkhorn_coupling = sinkhorn(a, b, C) 
    sinkhorn_distance = np.sum(C * sinkhorn_coupling)    
    precision = np.abs(sinkhorn_distance - exact_dist) / exact_dist
    return {'precision': precision}


experiment = Experiment("Sinkhorn precision test", ot_problems, run_function=conduct_precisoin_experiment)
results = experiment.run_experiment()
