import jax
jax.config.update("jax_enable_x64", True)

import time
from uot.algorithms.lp import pot_lp
from uot.algorithms.sinkhorn import jax_sinkhorn
from uot.core.experiment import generate_data_problems, get_problemset

# problem = get_problemset((3, "3dmesh", 1024))[0]
# print(problem)


problems = get_problemset(('distribution', "gaussian|gamma|beta|cauchy", 2048), number=45)

for problem in problems:
    print(problem)

