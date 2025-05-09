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
from uot.core.experiment import run_experiment
from uot.core.analysis import get_agg_table
from uot.core.suites import time_precision_experiment

epsilon_kwargs = [
    # {'epsilon': 100},
    # {'epsilon': 10},
    {'epsilon': 1},
    # {'epsilon': 1e-1},
    # {'epsilon': 1e-3},
    # {'epsilon': 1e-6},
    # {'epsilon': 1e-9},
]

solvers = {
    'pot-lp': (pot_lp, []),
    'lbfgs': (lbfgs_ot, epsilon_kwargs),
    'jax-sinkhorn': (jax_sinkhorn, epsilon_kwargs),
    'grad-ascent': (gradient_ascent, epsilon_kwargs),
}

# algorithms that use jax jit 
jit_algorithms = [
    'jax-sinkhorn', 'optax-grad-ascent', 'lbfgs'
]

problemset_names = [
    (1, "gamma", 32),
    # (1, "gamma", 64),
    # (1, "gamma", 256),
    # (1, "gamma", 512),
    # (1, "gamma", 1024),
    # (1, "gamma", 2048),

    # (1, "gaussian", 32),
    # (1, "gaussian", 64),
    # (1, "gaussian", 256),
    # (1, "gaussian", 512),
    # (1, "gaussian", 1024),
    # (1, "gaussian", 2048),

    # (1, "beta", 32),
    # (1, "beta", 64),
    # (1, "beta", 256),
    # (1, "beta", 512),
    # (1, "beta", 1024),
    # (1, "beta", 2048),

    # (1, "gaussian|gamma|beta|cauchy", 32),
    # (1, "gaussian|gamma|beta|cauchy", 64),
    # (1, "gaussian|gamma|beta|cauchy", 128),
    # (1, "gaussian|gamma|beta|cauchy", 256),
    # (1, "gaussian|gamma|beta|cauchy", 512),
    # (1, "gaussian|gamma|beta|cauchy", 1024),
    # (1, "gaussian|gamma|beta|cauchy", 2048),

    # (2, "WhiteNoise", 32),
    # (2, "CauchyDensity", 32),
    # (2, "GRFmoderate", 32),
    # (2, "GRFrough", 32),
    # (2, "GRFsmooth", 32),
    # (2, "LogGRF", 32),
    # (2, "LogitGRF", 32),
    # (2, "MicroscopyImages", 32),
    # (2, "Shapes", 32),
    # (2, "ClassicImages", 64),

    # (3, "3dmesh", 1024),
    # (3, "3dmesh", 2048),
]

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


df = run_experiment(experiment=time_precision_experiment, 
                    problemsets_names=problemset_names,
                    solvers=selected_solvers,
                    jit_algorithms=jit_algorithms,
                    folds=args.folds)


if args.save:
    df.to_csv(f"results/result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

if args.show:
    print(df)

if args.show_agg:
    dfs = [df[df.name == name] for name in df.name.unique()]

    for df in dfs:
        print("Solver", df.name.iloc[0])
        print(get_agg_table(df, ['time']))