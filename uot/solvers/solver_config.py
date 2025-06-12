from dataclasses import dataclass, field
from uot.solvers.base_solver import BaseSolver
from typing import List, Dict, Any


@dataclass
class SolverConfig:
    name: str
    solver: BaseSolver
    param_grid: List[Dict[str, Any]] = field(default_factory=list)
    is_jit: bool = False
