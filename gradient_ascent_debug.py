import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

from algorithms.gradient_ascent import gradient_ascent
from uot.experiment import run_experiment
from uot.suites import time_precision_suite

problemset_names = [
    "32 1D gamma",
]

solvers = {
    'optax-grad-ascent': gradient_ascent
}

results = run_experiment(time_precision_suite, problemset_names, solvers)