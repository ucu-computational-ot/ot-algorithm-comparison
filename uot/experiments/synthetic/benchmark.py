import os
import yaml
import argparse

from uot.experiments.runner import run_pipeline
from uot.utils.yaml_helpers import load_solvers, load_problems, load_experiment
from uot.utils.exceptions import InvalidConfigurationException
from uot.utils.logging import logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run time precision experiments.")

    parser.add_argument(
        "--save",
        help="Flag to save the results to a CSV file."
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help="Number of folds for cross-validation or repeated experiments."
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to a configuration file with experiment parameters."
    )

    parser.add_argument(
        "--export",
        type=str,
        default="gaussian_toy_results.csv",
        help="Path to export the results CSV file."
    )

    parser.add_argument(
        "--progress",
        type=bool,
        default=False,
        help="Show progress bar."
    )

    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    
    try:
        experiment = load_experiment(config=config)
        solver_configs = load_solvers(config=config)
        problems_iterators = load_problems(config=config)
    except InvalidConfigurationException as ex:
        print(ex)
        return


    df = run_pipeline(
        experiment=experiment,
        solvers=solver_configs,
        iterators=problems_iterators,
        folds=args.folds,
        progress=args.progress,
    )

    logger.info(f"Exporting results to {args.export}")
    os.makedirs(os.path.dirname(args.export), exist_ok=True)
    df.to_csv(args.export, index=False)


if __name__ == "__main__":
    main()
