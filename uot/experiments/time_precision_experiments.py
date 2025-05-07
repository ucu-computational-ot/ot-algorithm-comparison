import jax
jax.config.update("jax_enable_x64", True)

import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from uot.algorithms.sinkhorn import jax_sinkhorn
from uot.algorithms.gradient_ascent import gradient_ascent
from uot.algorithms.lbfgs import lbfgs_ot
from uot.algorithms.lp import pot_lp
from uot.core.experiment import run_experiment, generate_data_problems, get_problemset
from uot.core.analysis import get_agg_table
from uot.core.suites import time_precision_experiment, time_experiment

epsilon_kwargs = [
    {'epsilon': 100},
    {'epsilon': 10},
    {'epsilon': 1},
    {'epsilon': 1e-1},
    {'epsilon': 1e-3},
    {'epsilon': 1e-6},
    {'epsilon': 1e-9},
]

solvers = {
    'pot-lp': (pot_lp, []),
    'lbfgs': (lbfgs_ot, epsilon_kwargs),
    'jax-sinkhorn': (jax_sinkhorn, epsilon_kwargs),
    'optax-grad-ascent': (gradient_ascent, epsilon_kwargs),
}

# algorithms that use jax jit 
jit_algorithms = [
    'jax-sinkhorn', 'optax-grad-ascent', 'lbfgs'
]

problemset_names_1D = [
    "32 1D gamma",
    # "64 1D gamma",
    # "256 1D gamma",
    # "512 1D gamma",
    # "1024 1D gamma",
    # "2048 1D gamma",

    # "32 1D gaussian",
    # "64 1D gaussian",
    # "256 1D gaussian",
    # "512 1D gaussian",
    # "1024 1D gaussian",
    # "2048 1D gaussian",

    # "32 1D beta",
    # "64 1D beta",
    # "256 1D beta",
    # "512 1D beta",
    # "1024 1D beta",
    # "2048 1D beta",

    # '32 1D gaussian|gamma|beta|cauchy',
    # '64 1D gaussian|gamma|beta|cauchy',
    # '128 1D gaussian|gamma|beta|cauchy',
    # '256 1D gaussian|gamma|beta|cauchy',
    # '512 1D gaussian|gamma|beta|cauchy',
    # '1024 1D gaussian|gamma|beta|cauchy',
    # '2048 1D gaussian|gamma|beta|cauchy',
]


problem_sets_names_2D = [
    # ('WhiteNoise', 32),
    # ('CauchyDensity', 32),
    # ('GRFmoderate', 32),
    # ('GRFrough', 32),
    # ('GRFsmooth', 32),
    # ('LogGRF', 32),
    # ('LogitGRF', 32),
    # ('MicroscopyImages', 32),
    # ('Shapes', 32),
    # ('ClassicImages', 64)
]

problem_sets_3D = [
    # generate_3d_mesh_problems(1024, num_meshes=10),
    # generate_3d_mesh_problems(2048, num_meshes=10),
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


parser = argparse.ArgumentParser(description="Run time precision experiments.")
parser.add_argument(
    "--algorithms",
    nargs="+",
    default=list(solvers.keys()),
    help="List of algorithms to run experiments on. Use 'all' to run on all algorithms."
)
parser.add_argument(
    "--save",
    action="store_true",
    help="Flag to save the results to a CSV file."
)
parser.add_argument(
    "--show_agg",
    action="store_true",
    help="Flag to display the aggregated results table."
)
parser.add_argument(
    "--show",
    action="store_true",
    help="Flag to display the full results dataframe."
)
parser.add_argument(
    "--folds",
    type=int,
    default=5,
    help="Number of folds for cross-validation or repeated experiments."
)

args = parser.parse_args()

if "all" in args.algorithms:
    selected_solvers = solvers
else:
    selected_solvers = {key: solvers[key] for key in args.algorithms if key in solvers}

problems = problems * args.folds


df = run_experiment(experiment=time_experiment, problems=problems, solvers=selected_solvers, jit_algorithms=jit_algorithms)


if args.save:
    df.to_csv(f"results/result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

if args.show:
    print(df)

if args.show_agg:
    dfs = [df[df.name == name] for name in df.name.unique()]

    for df in dfs:
        print("Solver", df.name.iloc[0])
        print(get_agg_table(df, ['time']))