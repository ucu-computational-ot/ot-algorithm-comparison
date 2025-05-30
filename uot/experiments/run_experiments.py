import jax
jax.config.update("jax_enable_x64", True)

import os
import yaml
import argparse
import importlib

from datetime import datetime
from collections import namedtuple
from uot.core.experiment import run_experiment
from uot.core.suites import time_precision_experiment

Solver = namedtuple('Solver', ['function', 'params', 'is_jit'])

def load_solvers(config: dict) -> tuple[list, list]:
    solvers_configs = config['solvers']
    params = config['params']
    solvers = {}

    for solver_name, solver_config in solvers_configs.items():
        function = solver_config['function']
        is_jit = solver_config['jit']
        solver_params = solver_config['params']

        function = function.split('.')
        module, function = function[:-1], function[-1]
        module = '.'.join(module)

        mod = importlib.import_module(module)

        function = getattr(mod, function)
        solver_params = params[solver_params] if solver_params else []

        solvers[solver_name] = Solver(function=function, params=solver_params, is_jit=is_jit)

    return solvers 


parser = argparse.ArgumentParser(description="Run time precision experiments.")
parser.add_argument(
    "--algorithms",
    nargs="+",
    default=['all'],
    help="List of algorithms to run experiments on. Use 'all' to run on all algorithms."
)

parser.add_argument(
    "--save",
    help="Flag to save the results to a CSV file."
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

parser.add_argument(
    "--config",
    type=str,
    help="Path to a configuration file with experiment parameters."
)

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

try:
    problemset_names = config['problemsets']
    problemset_names = map(lambda x: x.split('-'), problemset_names)
    problemset_names = list(map(lambda x: (x[0], x[1], int(x[2])), problemset_names))
except KeyError:
    raise KeyError("The configuration file must contain a 'problemsets' key.")


solvers = load_solvers(config)


if "all" in args.algorithms:
    selected_solvers = solvers
else:
    selected_solvers = {key: solvers[key] for key in args.algorithms if key in solvers}


df = run_experiment(experiment=time_precision_experiment, 
                    problemsets_names=problemset_names,
                    solvers=selected_solvers,
                    folds=args.folds)


if args.save:
    export_filename = f"results/experiments_raw/result_{args.save}.csv"
    if os.path.exists(export_filename):
        export_filename = f"results/experiments_raw/result_{args.save}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv" 

    df.to_csv(export_filename)

if args.show:
    print(df)
