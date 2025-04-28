import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)
import time
from algorithms.gradient_ascent import gradient_ascent
from algorithms.sinkhorn import jax_sinkhorn, ott_jax_sinkhorn
from uot.experiment import run_experiment, generate_two_fold_problems, get_problemset
from uot.dataset import generate_gamma_pdf, generate_1d_gaussians_ds
from uot.analysis import get_agg_table
from uot.suites import time_precision_suite
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

problemset_names = [
    # "32 1D gaussian",
    "128 1D gaussian",
        # "256 1D gaussian",
    # "512 1D gaussian",
    # "1024 1D gaussian"
]


problem_sets = [
    get_problemset(problemset_name) for problemset_name in problemset_names
]
x = np.linspace(-5, 5, 1024)
problems = [problem for problem_set in problem_sets for problem in problem_set]


with tqdm(total=len(problems), desc="Running experiments") as pbar:
    progress_callback = lambda: pbar.update(1)
    result = time_precision_suite.run_suite("test run", problems, solver=ott_jax_sinkhorn, progress_callback=progress_callback)
result.df.drop(index=0, inplace=True)
result.display_agg()


