import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from uot.dataset import generate_1d_gaussians_ds, generate_2d_gaussians_ds, generate_3d_gaussians_ds
from algorithms.sinkhorn import sinkhorn
from uot.experiment import OTProblem, Experiment, get_exact_solution, generate_two_fold_problems 


x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
z = np.linspace(-10, 10, 100)
x2, y2 = np.meshgrid(x, y)
x, y, z = np.meshgrid(x, y, z)


gaussians_2d = generate_2d_gaussians_ds(x2, y2)
gaussians_3d = generate_3d_gaussians_ds(x, y, z)

gaussians_3d[3].plot()
# ot_problems = generate_two_fold_problems([x, y], gaussians)

# def conduct_precisoin_experiment(ot_problem: OTProblem):
#     a, b, C = ot_problem.a, ot_problem.b, ot_problem.C
#     _, exact_dist = get_exact_solution(a, b, C)

#     sinkhorn_coupling = sinkhorn(a, b, C) 
#     sinkhorn_distance = np.sum(C * sinkhorn_coupling)    
#     precision = np.abs(sinkhorn_distance - exact_dist) / exact_dist
#     return {'precision': precision}


# experiment = Experiment("Sinkhorn precision test", ot_problems, run_function=conduct_precisoin_experiment)
# results = experiment.run_experiment()
