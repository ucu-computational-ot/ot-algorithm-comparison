import pandas as pd
from tqdm import tqdm

from uot.experiments.experiment import Experiment
from uot.solvers.solver_config import SolverConfig
from uot.utils.logging import logger


def run_color_transfer_pipeline(
    experiment: Experiment,
    solvers: list[SolverConfig],
    problems: list,
    *,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Specialized runner for the color-transfer benchmarks.

    Avoids repeatedly deep-copying problem instances and makes sure solver
    parameters are written alongside every metrics row.
    """
    if not problems:
        logger.warning("No problems provided to the color transfer runner.")
        return pd.DataFrame()

    total_configs = sum(len(cfg.param_grid) if cfg.param_grid else 1 for cfg in solvers)
    total_runs = total_configs * len(problems)
    pbar = tqdm(total=total_runs, desc="Color transfer") if progress else None

    results: list[pd.DataFrame] = []

    def _progress_callback(n: int = 1):
        if pbar:
            pbar.update(n)

    for cfg in solvers:
        param_list = cfg.param_grid or [{}]
        for param_kwargs in param_list:
            if cfg.is_jit and problems:
                logger.debug(f"Warming up JIT for {cfg.name} with params {param_kwargs}")
                warm_df = experiment.run_on_problems(
                    problems=[problems[0]],
                    solver=cfg.solver,
                    progress_callback=None,
                    use_cost_matrix=cfg.use_cost_matrix,
                    **param_kwargs,
                )
                if not warm_df.empty:
                    logger.debug("Discarding warm-up results for JIT compilation.")
                if hasattr(problems[0], "free_memory"):
                    problems[0].free_memory()

            logger.info("Running %s with params %s", cfg.name, param_kwargs)
            df_res = experiment.run_on_problems(
                problems=problems,
                solver=cfg.solver,
                progress_callback=_progress_callback if pbar else None,
                use_cost_matrix=cfg.use_cost_matrix,
                **param_kwargs,
            )
            if df_res.empty:
                logger.warning(
                    "Solver %s with params %s produced no results.",
                    cfg.name,
                    param_kwargs,
                )
                continue

            df_res["name"] = cfg.name
            for key, value in param_kwargs.items():
                df_res[key] = value

            results.append(df_res)

            for prob in problems:
                if hasattr(prob, "free_memory"):
                    prob.free_memory()

    if pbar:
        pbar.close()

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True, sort=False)
