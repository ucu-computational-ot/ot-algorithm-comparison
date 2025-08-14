import jax
jax.config.update("jax_enable_x64", True)

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import jax.numpy as jnp
from pandas import DataFrame
import datetime
import logging

import uot.experiments.real_data.color_transfer.color_transfer_metrics as ct_metrics
from uot.utils.yaml_helpers import load_solvers, load_experiment
from uot.utils.types import ArrayLike
from uot.solvers.solver_config import SolverConfig
from uot.experiments.experiment import Experiment
from uot.experiments.runner import run_pipeline
from uot.utils.logging import logger
from uot.data.dataset_loader import load_matrix_as_color_grid
from uot.experiments.real_data.color_transfer.image_data import ImageData
from uot.experiments.real_data.color_transfer.utils import (
    get_image_problems,
    get_param_columns,
    im2mat,
    load_config_info,
    mat2im,
    match_shape,
    name_to_tuple,
    sample_image_pairs,
)


if os.environ.get('DEBUG', False):
    logger.setLevel(logging.DEBUG)


def transport_image(
    P: ArrayLike,
    source_image: ArrayLike,
    source_palette: ArrayLike,
    target_palette: ArrayLike,
    batch_size: int = 1024
) -> np.ndarray:
    """Transforms the image using transport plan. Returns NumPy array."""

    is_jax = isinstance(P, jax.Array) or hasattr(P, "device_buffer")
    lib = jnp if is_jax else np

    P_normalized = P / lib.sum(P, axis=1, keepdims=True)
    P_normalized = lib.nan_to_num(P_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    projected_palette = P_normalized @ target_palette

    if is_jax:
        return np.asarray(_transform_jax(source_image, source_palette, projected_palette))
    else:
        return _transform_numpy(source_image, source_palette, projected_palette, batch_size)
    
@jax.jit
def _transform_jax(source_image, source_palette, projected_palette):
    def transform_pixel(pixel):
        dists = jnp.linalg.norm(pixel - source_palette, axis=1)
        closest = jnp.argmin(dists)
        return projected_palette[closest]
    return jax.vmap(transform_pixel)(source_image)

def _transform_numpy(source_image, source_palette, projected_palette, batch_size):
    n_pixels = source_image.shape[0]
    transformed_chunks = []

    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        batch = source_image[start:end]

        diffs = batch[:, None, :] - source_palette[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        closest_bins = np.argmin(dists, axis=1)

        transformed_batch = projected_palette[closest_bins]
        transformed_chunks.append(transformed_batch)

    return np.concatenate(transformed_chunks, axis=0)


def perform_experiments(batch_size: int, data: dict[str, ImageData], image_pairs: set[tuple], solver_configs: list[SolverConfig], experiment: Experiment)-> tuple:
    """Carry out the experiments specified in the config"""

    results = {}
    logger.info("Running color transfer experiments...")


    output = run_pipeline(
        experiment=experiment,
        solvers=solver_configs,
        iterators=[get_image_problems(data, image_pairs)],
        folds=2,
        progress=True
    )

    param_cols = get_param_columns(output)
    output = output.drop_duplicates(subset=['dataset', 'name'] + param_cols, keep='last')

    logger.info(f"Transporting {len(output)} images...")

    for _, row in output.iterrows():
        problem_name = row['dataset']
        solver_name = row['name']

        key = name_to_tuple(problem_name)

        if key not in results:
            results[key] = {}

        if solver_name not in results[key]:
            results[key][solver_name] = {}
        
        param_kwargs = {col: row[col] for col in param_cols if col in row}

        param_key = frozenset(param_kwargs.items())

        try:
            P = row['transport_plan']
        except KeyError:
            logger.error(f"Transport plan not found for {problem_name} with solver {solver_name}. Make sure the experiment is configured to output it.")
            exit(1)

        img = im2mat(data[key[0]].get_image(use_jax=isinstance(P, jax.Array)))
        source_points = data[key[0]].get_grid(use_jax=isinstance(P, jax.Array)).to_discrete()[0]
        target_points = data[key[1]].get_grid(use_jax=isinstance(P, jax.Array)).to_discrete()[0]

        results[key][solver_name][param_key] = transport_image(
            P, img, source_points, target_points, batch_size
        )

    logger.info("Color transfer experiments completed, calculating metrics...")
    
    return output, results


def calculate_histogram_metrics(output: DataFrame, results: dict, data: dict[str, ImageData], param_columns: list[str], bin_num: int = 64) -> DataFrame:
    """Calculate histogram metrics for the color transfer experiments"""
    logger.info("Calculating histogram metrics...")

    wasserstein_distances = []
    kl_divergences = []

    for _, row in output.iterrows():
        problem_name = row['dataset']
        solver_name = row['name']

        key = name_to_tuple(problem_name)

        if key not in results or solver_name not in results[key]:
            logger.warning(f"Skipping {problem_name} with solver {solver_name} as results are not available.")
            continue

        param_kwargs = {col: row[col] for col in param_columns if col in row}

        img = results[key][solver_name][frozenset(param_kwargs.items())]

        target_grid = data[key[1]].get_grid()
        transferred_grid = load_matrix_as_color_grid(
            pixels=img,
            num_channels=3,
            bins_per_channel=bin_num,
            use_jax=False
        )

        wasserstein_distance = ct_metrics.compute_wasserstein_distance(
            source_grid=transferred_grid,
            target_grid=target_grid
        )

        kl_divergence = ct_metrics.compute_kl_divergence(
            source_grid=transferred_grid,
            target_grid=target_grid
        )

        wasserstein_distances.append(wasserstein_distance)
        kl_divergences.append(kl_divergence)
        
    output['wasserstein_distance'] = wasserstein_distances
    output['kl_divergence'] = kl_divergences

    return output


def calculate_quality_metrics(output: DataFrame, results: dict, data: dict[str, ImageData], param_columns: list[str]) -> DataFrame:
    """Compute SSIM, ΔE, and colorfulness metrics for transferred images."""
    logger.info("Calculating perceptual quality metrics...")

    ssims = []
    delta_es = []
    colorfulness_diffs = []

    for _, row in output.iterrows():
        problem_name = row['dataset']
        solver_name = row['name']
        key = name_to_tuple(problem_name)

        if key not in results or solver_name not in results[key]:
            logger.warning(f"Skipping {problem_name} with solver {solver_name} as results are not available.")
            continue

        param_kwargs = {col: row[col] for col in param_columns if col in row}
        transferred = mat2im(results[key][solver_name][frozenset(param_kwargs.items())], data[key[0]].get_image_shape())
        target = data[key[1]].get_image()
        source = data[key[0]].get_image()

        transferred = np.clip(transferred, 0, 1)
        target = np.clip(target, 0, 1)
        source = np.clip(source, 0, 1)

        target_resized = match_shape(target, transferred)

        assert source.shape == transferred.shape, f'{source.shape, transferred.shape}'

        ssims.append(ct_metrics.compute_ssim_metric(transferred, source))
        delta_es.append(ct_metrics.compute_delta_e(transferred, target_resized))

        c1 = ct_metrics.compute_colorfulness(transferred)
        c2 = ct_metrics.compute_colorfulness(target)
        colorfulness_diffs.append(np.abs(c1 - c2))

    output['ssim'] = ssims
    output['delta_e'] = delta_es
    output['colorfulness_diff'] = colorfulness_diffs

    return output


def calculate_spatial_metrics(output: DataFrame, results: dict, data: dict[str, ImageData], param_columns: list[str]) -> DataFrame:
    logger.info("Calculating spatial-structure metrics...")

    gradient_corrs = []
    sharpness_values = []

    for _, row in output.iterrows():
        problem_name = row['dataset']
        solver_name = row['name']
        key = name_to_tuple(problem_name)

        if key not in results or solver_name not in results[key]:
            logger.warning(f"Skipping {problem_name} with solver {solver_name} as results are not available.")
            continue

        param_kwargs = {col: row[col] for col in param_columns if col in row}
        img_transferred = results[key][solver_name][frozenset(param_kwargs.items())]
        img_transferred = mat2im(img_transferred, data[key[0]].get_image().shape)

        img_source = data[key[0]].get_image()

        gradient_corr = ct_metrics.compute_gradient_magnitude_correlation(img_transferred, img_source)
        sharpness_transferred = ct_metrics.compute_laplacian_variance(img_transferred)
        sharpness_source = ct_metrics.compute_laplacian_variance(img_source)

        gradient_corrs.append(gradient_corr)
        sharpness_values.append(sharpness_transferred - sharpness_source)

    output['gradient_correlation'] = gradient_corrs
    output['laplacian_sharpness_diff'] = sharpness_values

    return output


def export_data(output: DataFrame, results: dict, output_dir: str, data: dict[str, ImageData], param_columns: list[str], drop_columns: list[str]):
    """Export the results of the experiments to CSV and images"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(output_dir, f"color_transfer_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    img_dir = os.path.join(base_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(base_dir, "color_transfer_results.csv")

    if drop_columns:

        if any(col not in output.columns for col in drop_columns):

            wrong_columns = [col for col in drop_columns if col not in output.columns]
            logger.warning(f"Some columns to drop are not present in the output: {wrong_columns}")

        output = output.drop(columns=drop_columns)

    output.to_csv(csv_path, index=False)

    for _, row in output.iterrows():
        problem_name = row['dataset']
        solver_name = row['name']
        key = name_to_tuple(problem_name)

        if key not in results or solver_name not in results[key]:
            continue

        param_kwargs = {col: row[col] for col in param_columns if col in row}
        img = results[key][solver_name][frozenset(param_kwargs.items())]
        img = mat2im(img, data[key[0]].get_image_shape())
        img = np.clip(img, 0, 1)

        param_str = "_".join(f"{k}_{v}" for k, v in sorted(param_kwargs.items()))
        base_name = f"{problem_name.replace(' -> ', '_')}_{solver_name}_{param_str}.png"
        save_path = os.path.join(img_dir, base_name)

        plt.imsave(save_path, img)

    logger.info(f"Results exported to {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run color transfer via optimal transport")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a configuration file with experiment parameters."
    )

    args = parser.parse_args()

    logger.debug('Reading configuration...')
    with open(args.config) as file:
        config = yaml.safe_load(file)

    logger.debug('Loading configuration...')
    bin_num, batch_size, pair_num, images_dir, output_dir, rng_seed, drop_columns = load_config_info(config)
    logger.debug('Loading experiment...')
    experiment = load_experiment(config)

    logger.debug('Loading solvers...')
    solver_configs = load_solvers(config=config)
    ImageData.set_image_dir(images_dir)

    logger.debug('Sampling images pairs...')
    data, image_pairs = sample_image_pairs(images_dir, bin_num, pair_num, seed=rng_seed)

    logger.debug('Performing experiments...')
    output, results = perform_experiments(batch_size, data, image_pairs, solver_configs, experiment)
    param_columns = get_param_columns(output)

    output = calculate_histogram_metrics(output, results, data, param_columns, bin_num)
    output = calculate_quality_metrics(output, results, data, param_columns)
    output = calculate_spatial_metrics(output, results, data, param_columns)

    export_data(output, results, output_dir, data, param_columns, drop_columns)
