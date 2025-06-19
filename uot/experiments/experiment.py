import pandas as pd
from typing import List, Optional, Callable, Dict, Iterable
from uot.problems.base_problem import MarginalProblem
from uot.data.measure import BaseMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike
from uot.utils.logging import logger


class Experiment:
    def __init__(
            self,
            name: str,
            solve_fn: Callable[[MarginalProblem, BaseSolver, List[BaseMeasure],
                                List[ArrayLike]], Dict],
    ):
        """
        solve_fn: a function f(problem: MarginalProblem, solver: BaseSolver, **kwargs) -> metrics dict
        """
        self.name = name
        self.solve_fn = solve_fn

    def run_on_problems(
        self,
        problems: Iterable[MarginalProblem],
        solver: BaseSolver,
        progress_callback: Optional[Callable[[int], None]] = None,
        **solver_kwargs,
    ) -> pd.DataFrame:
        results = []
        for i, problem in enumerate(problems):
            marginals = problem.get_marginals()
            costs = problem.get_costs()
            try:
                metrics = self.solve_fn(
                    problem,
                    solver,
                    marginals,
                    costs,
                    **solver_kwargs,
                )
                metrics["status"] = "success"
                logger.info(f"Successfully run {
                            solver.__class__.__name__} with {solver_kwargs}")
            except Exception as e:
                logger.error(f"{solver.__qualname__} failed with error {e}")
                metrics = {
                    "status": "failed",
                    "exception": str(e),
                }
            df_row = problem.to_dict()
            df_row.update(metrics)
            results.append(df_row)
            if progress_callback:
                progress_callback(1)
        return pd.DataFrame(results)

    def run_single(self, problem: MarginalProblem, solver: BaseSolver, **solver_kwargs) -> Dict:
        marginals = problem.get_marginals()
        costs = problem.get_costs()
        return self.solve_fn(
            problem,
            solver,
            marginals,
            costs,
            **solver_kwargs,
        )
