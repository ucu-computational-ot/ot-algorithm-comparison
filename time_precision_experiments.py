import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

from algorithms.lp import pot_lp
from algorithms.sinkhorn import sinkhorn, ott_jax_sinkhorn, pot_sinkhorn
from algorithms.gradient_ascent import gradient_ascent
from uot.analysis import get_agg_table, display_all_metrics
from uot.experiment import run_experiment, get_problemset
from uot.suites import time_precision_suite

solvers = {
    # 'pot-lp': pot_lp,
    # 'pot-sinkhorn': pot_sinkhorn,
    # 'ott-jax-sinkhorn': ott_jax_sinkhorn,
    'jax-sinkhorn': sinkhorn,
    # 'optax-grad-ascent': gradient_ascent
}


# problemset_names = [
#     "32 1D gamma",
#     # "64 1D gamma",
#     # "128 1D gamma",
#     # "256 1D gamma",
#     # "512 1D gamma",
#     # "1024 1D gamma"
# ]

# results = run_experiment(time_precision_suite, problemset_names, solvers)