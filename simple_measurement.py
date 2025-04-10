import numpy as np
import pandas as pd
from functools import partial
from uot.experiment import ExperimentSuite, Experiment, generate_two_fold_problems, time_experiment, precision_experiment, memory_experiment
from uot.dataset import generate_1d_gaussians_ds, generate_2d_gaussians_ds
from algorithms.sinkhorn import sinkhorn, ott_jax_sinkhorn
from algorithms.lp import pot_lp
from tqdm import tqdm

x = np.linspace(-5, 5, 100)


def create_suite_for_algorithm(solver):
    suite = ExperimentSuite(experiments=[
        Experiment("Measure time", run_function=partial(time_experiment, solver=solver)),
        Experiment("Measure precision", run_function=partial(precision_experiment, solver=solver)),
        Experiment("Measure memory", run_function=partial(memory_experiment, solver=solver))
    ])
    return suite

own_sinkhorn_suite = create_suite_for_algorithm(sinkhorn)
ott_jax_sinkhorn_suite = create_suite_for_algorithm(ott_jax_sinkhorn)
pot_lp_suite = create_suite_for_algorithm(pot_lp)


GRID_1D = {
    '32': np.linspace(-5, 5, 32),
    '64': np.linspace(-5, 5, 64),
    '128': np.linspace(-5, 5, 256),
    '512': np.linspace(-5, 5, 512)
}

GRID_2D = {
    '4x4': np.meshgrid(np.linspace(-5, 5, 4), np.linspace(-5, 5, 4)),
    '8x8': np.meshgrid(np.linspace(-5, 5, 8), np.linspace(-5, 5, 8)),
    '16x16': np.meshgrid(np.linspace(-5, 5, 16), np.linspace(-5, 5, 16)),
    '32x32': np.meshgrid(np.linspace(-5, 5, 32), np.linspace(-5, 5, 32)),
}


problems_1d = [generate_two_fold_problems(grid, generate_1d_gaussians_ds(x), name=f"{dname} 1D Gaussians")
               for dname, grid in GRID_1D.items()]

problems_2d = [generate_two_fold_problems(grid, generate_2d_gaussians_ds(*grid), name=f"{dname} 2D Gaussians")
               for dname, grid in GRID_2D.items()]

problems =  problems_1d + problems_2d

sinkhorn_dfs = []
ott_jax_sinkhorn_dfs = []
pot_lp_dfs = []


with tqdm(total=len(problems) * 3, desc="Running experiments") as pbar:

    for ot_problems in problems: 
        sinkhorn_dfs.append(own_sinkhorn_suite.run_suite(ot_problems))
        pbar.update(1)
        ott_jax_sinkhorn_dfs.append(ott_jax_sinkhorn_suite.run_suite(ot_problems))
        pbar.update(1)
        pot_lp_dfs.append(pot_lp_suite.run_suite(ot_problems))
        pbar.update(1)


sinkhorn_df = pd.concat(sinkhorn_dfs)
ott_jax_sinkhorn_df = pd.concat(ott_jax_sinkhorn_dfs)
pot_lp_dfs = pd.concat(pot_lp_dfs)

sinkhorn_df.to_csv("results/sinkhorn.csv")
ott_jax_sinkhorn_df.to_csv("results/ott_jax_sinkhorn.csv")
pot_lp_dfs.to_csv("results/pot_lp_dfs.csv")