import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from functools import partial
from uot.dataset import generate_1d_gaussians_ds
from uot.experiment import Experiment, generate_two_fold_problems, precision_experiment
from algorithms.sinkhorn import ott_jax_sinkhorn
from memory_profiler import memory_usage, profile


x = np.linspace(-10, 10, 20)

gaussians_1d = generate_1d_gaussians_ds(x)
ot_problems = generate_two_fold_problems([x], gaussians_1d, name="test")



experiment = Experiment("Sinkhorn precision test",run_function=partial(precision_experiment, solver=ott_jax_sinkhorn))
results = experiment.run_experiment(ot_problems=ot_problems)

# print(results)
