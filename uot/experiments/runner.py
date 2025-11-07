import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from uot.solvers.solver_config import SolverConfig
from uot.experiments.experiment import Experiment
from uot.problems.base_problem import MarginalProblem
from collections.abc import Iterable
from uot.utils.logging import logger


def run_pipeline(
    experiment: Experiment,
    solvers: list[SolverConfig],
    iterators: list[Iterable[MarginalProblem]],
    folds: int = 1,
    progress: bool = True,
) -> pd.DataFrame:
    # 1) generate all problems
    logger.info("starting pipeline...")

    # 2) apply folds
    all_iterators = []
    for _ in range(folds):
        all_iterators += deepcopy(iterators)

    # 3) run each solver on params and problems
    results_list = []

    # count how many same problems may be used (re-runned)
    # for different solver's parameters
    problems_multiplicity = sum(
        len(cfg.param_grid) if cfg.param_grid else 1 for cfg in solvers)
    total_runs = problems_multiplicity * sum(len(it) for it in all_iterators)

    pbar = tqdm(total=total_runs,
                desc="Running experiments") if progress else None

    all_iterators = chain(*all_iterators)

    # copy is needed, because same problems instances
    # are needed to run with different solver configuration
    current_iterators = deepcopy(all_iterators)

    def progress_callback(n=1):
        if pbar:
            pbar.update(n)

    logger.info("Running experiments...")
    for cfg in solvers:
        params = cfg.param_grid
        for param_kwargs in params:
            if cfg.is_jit:
                logger.debug(f"Warming up JIT compilation for {cfg.name}")
                first_problem = next(deepcopy(current_iterators))
                experiment.run_on_problems(
                    problems=[first_problem],
                    solver=cfg.solver,
                    progress_callback=None,
                    **param_kwargs,
                )

            logger.info(f"Running set of problems on {cfg.solver}\
            with {param_kwargs}")
            description = f"{cfg.name}({param_kwargs})"
            if pbar:
                pbar.set_description(description)

            df_res = experiment.run_on_problems(
                problems=current_iterators,
                solver=cfg.solver,
                progress_callback=progress_callback if pbar else None,
                use_cost_matrix=cfg.use_cost_matrix,
                **param_kwargs,
            )

            if len(df_res) == 0:
                logger.warning("Result of run returned empty. Either the\
                    set of problems is empty or the solver and\
                    dataset is encorrectly configured.")
                continue

            df_res["name"] = cfg.name

            current_iterators = deepcopy(all_iterators)

            results_list.append(df_res)

    if pbar:
        pbar.close()
    logger.info("Composing results...")
    df = pd.concat(results_list, ignore_index=True, sort=False)
    return df
