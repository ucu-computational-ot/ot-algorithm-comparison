from uot.algorithms.lp import pot_lp
from uot.core.experiment import get_problemset

problem = get_problemset("1024 1D gaussian")[8]

a, b, C = problem.a, problem.b, problem.C
_, cost = pot_lp(a, b, C)

print("Exact cost", problem.exact_cost)
print("Recumputed", cost)

