import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
from uot.experiment import generate_two_fold_problems
from uot.suites import standard_suite
from algorithms.lp import pot_lp
from algorithms.sinkhorn import sinkhorn, ott_jax_sinkhorn, pot_sinkhorn
from algorithms.pdgd import pdhg_algorithm
from uot.dataset import generate_1d_gaussians_ds, generate_2d_gaussians_ds
from tqdm import tqdm

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

problems_1d = [generate_two_fold_problems(grid, generate_1d_gaussians_ds(grid), name=f"{dname} 1D Gaussians")
               for dname, grid in GRID_1D.items()]

problems_2d = [generate_two_fold_problems(grid, generate_2d_gaussians_ds(*grid), name=f"{dname} 2D Gaussians")
               for dname, grid in GRID_2D.items()]

problems =  problems_1d + problems_2d

problems = [problem for sublist in problems for problem in sublist]

with tqdm(total=4 * 3 * len(problems), desc="Running experiments") as pbar:
    progress_callback = lambda: pbar.update(1)

    own_sinkhorn_result = standard_suite.run_suite(name="JAX-Sinkhorn", ot_problems=problems,
                                          progress_callback=progress_callback, solver=sinkhorn)

    ott_jax_sinkhorn_result = standard_suite.run_suite(name="OTT-JAX-Sinkhorn", ot_problems=problems,
                                              progress_callback=progress_callback, solver=ott_jax_sinkhorn)

    pot_sinkhorn_result = standard_suite.run_suite(name="POT-Sinkhorn", ot_problems=problems,
                                          progress_callback=progress_callback, solver=pot_sinkhorn)

    pot_lp_result = standard_suite.run_suite(name="POT-LP", ot_problems=problems,
                                          progress_callback=progress_callback, solver=pot_lp)


own_sinkhorn_result.export("results/sinkhorn.csv")
ott_jax_sinkhorn_result.export("results/ott_jax_sinkhorn.csv")
pot_sinkhorn_result.export("results/pot_sinkhorn.csv")
pot_lp_result.export("results/pot_lp.csv")