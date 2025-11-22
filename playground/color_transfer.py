import jax
jax.config.update("jax_enable_x64", True)

import os
import matplotlib.pyplot as plt
import yaml
import datetime
import logging

from uot.utils.yaml_helpers import load_solvers, load_experiment
from uot.experiments.runner import run_pipeline
from uot.utils.logging import logger
from uot.experiments.real_data.color_transfer.image_data import ImageData
from uot.experiments.real_data.color_transfer.utils import (
    get_image_problems,
    load_config_info,
    sample_image_pairs,
)
from uot.experiments.real_data.color_transfer.experiment import ColorTransferExperiment


if os.environ.get('DEBUG', False):
    logger.setLevel(logging.DEBUG)


def main():
    with open('configs/color_transfer/adobe.yaml') as file:
        config = yaml.safe_load(file)
    
    print(config)

    bin_num, batch_size, pair_num, images_dir, rng_seed, drop_columns = load_config_info(config)
    output_dir = config['experiment'].get('output-dir', 'output/color_transfer')

    logger.debug('Loading solvers...')
    solver_configs = load_solvers(config=config)
    ImageData.set_image_dir(images_dir)

    logger.info('Sampling and loading images pairs...')
    data, image_pairs = sample_image_pairs(images_dir, bin_num, pair_num, seed=rng_seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(output_dir, f"color_transfer_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    experiment = ColorTransferExperiment(
        name=config['experiment']['name'],
        output_dir=base_dir,
        drop_columns=drop_columns,
    )
    results = run_pipeline(
        experiment=experiment,
        solvers=solver_configs,
        iterators=[get_image_problems(data, image_pairs)],
        folds=1,
        progress=True
    )

    csv_path = os.path.join(base_dir, "color_transfer_results.csv")
    results.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()