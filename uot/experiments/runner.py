import pandas as pd
from tqdm import tqdm
from typing import List
from uot.experiments.solver_config import SolverConfig
from uot.experiments.experiment import Experiment
from uot.problems.base_problem import MarginalProblem
from uot.problems.generators import ProblemGenerator
from uot.utils.logging import logger


def run_pipeline(
    experiment: Experiment,
    solvers: List[SolverConfig],
    generators: List[ProblemGenerator],
    folds: int = 1,
    progress: bool = True,
) -> pd.DataFrame:
    # 1) generate all problems
    logger.info(f"starting pipeline...")
    logger.info(f"generating problems...")
    all_problems: List[MarginalProblem] = []
    for generator in generators:
        all_problems.extend(generator.generate())
    # 2) apply folds
    all_problems = all_problems * folds
    # 3) run each solver on params and problems
    results_list = []
    # TODO: tqdm status bar
    total_runs = sum(len(cfg.param_grid)
                     for cfg in solvers) * len(all_problems)
    pbar = tqdm(total=total_runs,
                desc="Running experiments") if progress else None

    def progress_callback(n=1):
        if pbar:
            pbar.update(n)

    logger.info("Running experiments...")
    for cfg in solvers:
        params = cfg.param_grid
        for param_kwargs in params:
            description = f"{cfg.name}"
            if pbar:
                pbar.set_description(description)

            df_res = experiment.run_on_problems(
                problems=all_problems,
                solver=cfg.solver,
                progress_callback=progress_callback if pbar else None,
                **param_kwargs,
            )

            df_res["name"] = cfg.name
            for k, v in param_kwargs.items():
                df_res[k] = v

            if cfg.is_jit:
                # NOTE: basically, drop the first occurence filtering
                #       on ds and cfg name
                drop_idxs = []
                for dataset in df_res["dataset"].unique():
                    subset = df_res[
                        (df_res["dataset"] == dataset) & (
                            df_res["name"] == cfg.name)
                    ]
                    if not subset.empty:
                        drop_idxs.append(subset.index[0])
                df_res = df_res.drop(drop_idxs, axis=0)

            results_list.append(df_res)

    if pbar:
        pbar.close()
    logger.info("Composing results...")
    df = pd.concat(results_list, ignore_index=True, sort=False)
    return df
