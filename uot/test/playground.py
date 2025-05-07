import jax
jax.config.update("jax_enable_x64", True)

import time
from uot.algorithms.lp import pot_lp
from uot.algorithms.sinkhorn import jax_sinkhorn
from uot.core.experiment import generate_data_problems, get_problemset

problem = get_problemset((3, "3dmesh", 1024))[0]
print(problem)

# N = 10
# a, b, C = problem.to_jax_arrays()

# total = 0
# for i in range(N):
#     print(i)
#     start = time.perf_counter()
#     _, cost = pot_lp(a, b, C)
#     end = time.perf_counter()
#     total += end - start

# print("LP:",  1000 * total / N)

# total = 0
# for i in range(N):
#     print(i)
#     start = time.perf_counter()
#     _, cost = jax_sinkhorn(a, b, C)
#     end = time.perf_counter()
#     total += end - start
# print("JAX:",  1000 * total / N)

