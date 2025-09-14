from dataclasses import dataclass, field
from uot.solvers.base_solver import BaseSolver
from typing import Any


@dataclass
class SolverConfig:
    name: str
    solver: BaseSolver
    param_grid: list[dict[str, Any]] = field(default_factory=list)
    is_jit: bool = False
    use_cost_matrix: bool = True
