import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from uot.dataset import generate_1d_gaussians_ds
from algorithms.sinkhorn import sinkhorn
from uot.experiment import OTProblem, Experiment, get_q_const, get_exact_solution


x = np.linspace(-6, 6, 100)
gaussians = generate_1d_gaussians_ds(x)

fig, axes = plt.subplots(len(gaussians), 1, figsize=(10, 6 * len(gaussians)))

for i, (gaussian, ax) in enumerate(zip(gaussians, axes)):
    ax.plot(x, gaussian, label=f"Gaussian {i+1}")
    ax.set_title(f"1D Gaussian Distribution {i+1}")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


# gaussians = generate_1d_gaussians(x)
# ot_problems = []

# for (a, b) in it.combinations(gaussians, 2):
#     C = get_q_const(x, x)
#     ot_problems.append(OTProblem("Transport two gaussians", a, b, C))


# def conduct_precisoin_experiment(ot_problem: OTProblem):
#     a, b, c = ot_problem.a, ot_problem.b, ot_problem.C
#     _, exact_dist = get_exact_solution(a, b, C)

#     sinkhorn_coupling = sinkhorn(a, b, C) 
#     sinkhorn_distance = np.sum(C * sinkhorn_coupling)    
#     precision = np.abs(sinkhorn_distance - exact_dist) / exact_dist
#     return {'precision': precision}


# experiment = Experiment("Sinkhorn precision test", ot_problems, run_function=conduct_precisoin_experiment)
# results = experiment.run_experiment()
