import jax
jax.config.update("jax_enable_x64", True)

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from uot.algorithms.sinkhorn import jax_sinkhorn, ott_jax_sinkhorn
from uot.algorithms.gradient_ascent import gradient_ascent
from uot.algorithms.lbfgs import dual_lbfgs
from uot.algorithms.lp import pot_lp
from uot.algorithms.col_gen import col_gen
from uot.core.experiment import run_experiment, generate_data_problems, generate_3d_mesh_problems, get_problemset
from uot.core.analysis import get_agg_table
from uot.core.suites import time_precision_suite

solvers = {
    'pot-lp': pot_lp,
    'second-order-lbfgs': dual_lbfgs,
    'ott-jax-sinkhorn': ott_jax_sinkhorn,
    'jax-sinkhorn': jax_sinkhorn,
    'optax-grad-ascent': gradient_ascent,
}

# algorithms that use jax jit 
jit_algorithms = [
    'ott-jax-sinkhorn', 'jax-sinkhorn', 'optax-grad-ascent'
]

problemset_names_1D = [
    "32 1D gamma",
    "64 1D gamma",
    "256 1D gamma",
    "512 1D gamma",
    "1024 1D gamma",
    "2048 1D gamma",

    "32 1D gaussian",
    "64 1D gaussian",
    "256 1D gaussian",
    "512 1D gaussian",
    "1024 1D gaussian",
    "2048 1D gaussian",

    "32 1D beta",
    "64 1D beta",
    "256 1D beta",
    "512 1D beta",
    "1024 1D beta",
    "2048 1D beta",

    '32 1D gaussian|gamma|beta|cauchy',
    '64 1D gaussian|gamma|beta|cauchy',
    '128 1D gaussian|gamma|beta|cauchy',
    '256 1D gaussian|gamma|beta|cauchy',
    '512 1D gaussian|gamma|beta|cauchy',
    '1024 1D gaussian|gamma|beta|cauchy',
    '2048 1D gaussian|gamma|beta|cauchy',
    
]


problem_sets_names_2D = [
    ('WhiteNoise', 32),
    ('CauchyDensity', 32),
    ('GRFmoderate', 32),
    ('GRFrough', 32),
    ('GRFsmooth', 32),
    ('LogGRF', 32),
    ('LogitGRF', 32),
    ('MicroscopyImages', 32),
    ('Shapes', 32),
    ('ClassicImages', 32)
]

problem_sets_3D = [
    generate_3d_mesh_problems(1024, num_meshes=10),
    generate_3d_mesh_problems(2048, num_meshes=10),
]

problem_sets = [get_problemset(name) for name in problemset_names_1D] + \
               [generate_data_problems(*name) for name in problem_sets_names_2D] + \
               problem_sets_3D

problems = [problem for problemset in problem_sets 
                    for problem in problemset]


for problem in problems:
    source_distribution = problem.source_measure.distribution
    target_distribution = problem.target_measure.distribution
    
    if np.any(np.logical_or(np.isnan(source_distribution), np.isinf(source_distribution))):
        print(f"Detected problem in: {problem}")
        print("Source distribution contains NaN or Inf")
        print(source_distribution)
        sys.exit()
     
    if np.any(np.logical_or(np.isnan(target_distribution), np.isinf(target_distribution))):
        print(f"Detected problem in: {problem}")
        print("Source distribution contains NaN or Inf")
        print(target_distribution)
        sys.exit()

results = run_experiment(suite=time_precision_suite, problems=problems, solvers=solvers)

result_df = pd.concat(list(results.values()))

for dataset in result_df.dataset.unique():
    for algorithm_name in jit_algorithms:
        algorithm_results = result_df[(result_df.dataset == dataset) & (result_df.name == algorithm_name)]
        if len(algorithm_results):
            result_df.drop(algorithm_results.index[0], inplace=True)

# result_df.to_csv(f"results/result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
print(get_agg_table(result_df, ['cost_rerr', 'coupling_avg_err']))