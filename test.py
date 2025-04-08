import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from uot.dataset import generate_1d_gaussians_ds, generate_2d_gaussians_ds, generate_3d_gaussians_ds
from algorithms.sinkhorn import sinkhorn
from uot.experiment import OTProblem, Experiment, ExperimentSuite, generate_two_fold_problems 


x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
x, y = np.meshgrid(x, y)


gaussians_2d = generate_2d_gaussians_ds(x, y)
ot_problems = generate_two_fold_problems([x, y], gaussians_2d)


def conduct_precisoin_experiment(ot_problem: OTProblem):
    return {'precision': np.random.randn()}

def conduct_dummy_experiment(ot_problem: OTProblem):
    return {'dummy_variable': 1}

ot_problems[0].kwargs = {'eps': 1}

experiment = Experiment("Sinkhorn precision test", run_function=conduct_precisoin_experiment)
second_expetiment = Experiment("Dummy experiment", run_function=conduct_dummy_experiment)
experiment_suite = ExperimentSuite([experiment, second_expetiment])
results = experiment_suite.run_suite(ot_problems, njobs=2)
print(results)
