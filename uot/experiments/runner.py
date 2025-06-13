import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from typing import List
from itertools import chain 
from uot.solvers.solver_config import SolverConfig
from uot.experiments.experiment import Experiment
from uot.problems.base_problem import MarginalProblem
from uot.problems.iterator import ProblemIterator
from uot.utils.logging import logger


def run_pipeline(
    experiment: Experiment,
    solvers: List[SolverConfig],
    iterators: List[ProblemIterator],
    folds: int = 1,
    progress: bool = True,
) -> pd.DataFrame:
    # 1) generate all problems
    logger.info(f"starting pipeline...")
    logger.info(f"generating problems...")

    # 2) apply folds
    all_iterators = []
    for _ in range(folds):
        all_iterators += deepcopy(iterators)
    
    # 3) run each solver on params and problems
    results_list = []
    # TODO: tqdm status bar

    # count how many same problems may be used (re-runned) 
    # for different solver's parameters
    problems_multiplicity = sum(len(cfg.param_grid) if cfg.param_grid else 1 for cfg in solvers)
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
            description = f"{cfg.name}"
            if pbar:
                pbar.set_description(description)
            
            df_res = experiment.run_on_problems(
                problems=current_iterators,
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

            current_iterators = deepcopy(all_iterators)

            results_list.append(df_res)

    if pbar:
        pbar.close()
    logger.info("Composing results...")
    df = pd.concat(results_list, ignore_index=True, sort=False)
    return df
