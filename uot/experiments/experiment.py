import pandas as pd
from collections.abc import Callable, Iterable
from uot.problems.base_problem import MarginalProblem
from uot.data.measure import BaseMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike
from uot.utils.logging import logger


class Experiment:
    def __init__(
            self,
            name: str,
            solve_fn: Callable[[MarginalProblem, BaseSolver, list[BaseMeasure],
                                list[ArrayLike]], dict],
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
        progress_callback: Callable[[int], None] | None = None,
        use_cost_matrix: bool = True,
        **solver_kwargs,
    ) -> pd.DataFrame:
        results = []
        for i, problem in enumerate(problems):
            marginals = problem.get_marginals()
            costs: list[ArrayLike] = problem.get_costs() if use_cost_matrix else []
            logger.info(f"Starting {solver.__name__} with {solver_kwargs} on {problem}")
            try:
                metrics = self.solve_fn(
                    problem,
                    solver,
                    marginals,
                    costs,
                    **solver_kwargs,
                )
                metrics["status"] = "success"
                logger.info(f"Successfully finished {solver.__name__} with {solver_kwargs}")
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

    def run_single(self, problem: MarginalProblem, solver: BaseSolver, **solver_kwargs) -> dict:
        marginals = problem.get_marginals()
        costs = problem.get_costs()
        return self.solve_fn(
            problem,
            solver,
            marginals,
            costs,
            **solver_kwargs,
        )
