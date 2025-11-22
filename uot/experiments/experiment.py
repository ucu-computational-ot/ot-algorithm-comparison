import pandas as pd
import jax.numpy as jnp
from collections.abc import Callable, Iterable
from uot.problems.base_problem import MarginalProblem
from uot.data.measure import BaseMeasure
from uot.solvers.base_solver import BaseSolver
from uot.utils.types import ArrayLike
from uot.utils.logging import logger
from uot.utils.instantiate_solver import instantiate_solver
from uot.solvers.gradient_ascent._smith_best_lr import best_lr as _best_lr


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

    def _run_lr_finder(self, solver: BaseSolver,
                       marginals: list[BaseMeasure],
                       costs: list[ArrayLike],
                       **solver_kwargs) -> float:
        lrs = []
        losses = []
        if not hasattr(solver, "find_lr"):
            raise RuntimeError(f"Solver {solver.__class__.__name__} has no `find_lr` method")
        lrs, losses = solver.find_lr(
            marginals=marginals,
            costs=costs,
            **solver_kwargs,
        )
        # just set the lr that gives the best dual value
        lr = lrs[jnp.argmax(jnp.array(losses))]
        # lr = _best_lr(
        #     lrs=jnp.array(lrs),
        #     losses=jnp.array(losses),
        # )
        return lr

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
            try:
                solver_init_kwargs = solver_kwargs or {}
                if "learning_rate" in solver_init_kwargs and solver_init_kwargs["learning_rate"] == "auto":
                    logger.info(f"Finding learning rate for {solver.__name__} on {problem}")
                    lr = self._run_lr_finder(
                        solver=solver,
                        marginals=marginals,
                        costs=costs,
                        **solver_init_kwargs,
                    )
                    solver_init_kwargs["learning_rate"] = lr
                    logger.info(f"Found learning rate: {lr:.3e}")

                instance = instantiate_solver(solver_cls=solver, init_kwargs=solver_init_kwargs)

                logger.info(f"Starting {solver.__name__} with {solver_kwargs} on {problem}")
                metrics = self.solve_fn(
                    problem,
                    instance,
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
            metrics["problem_index"] = i
            for solver_key, solver_value in solver_init_kwargs.items():
                metrics[solver_key] = solver_value
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
