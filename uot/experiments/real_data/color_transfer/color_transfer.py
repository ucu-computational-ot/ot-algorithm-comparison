import jax
jax.config.update("jax_enable_x64", True)

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import yaml
import jax.numpy as jnp

from uot.utils.yaml_helpers import load_solvers
from uot.utils.costs import cost_euclid
from uot.utils.types import ArrayLike
from uot.solvers.solver_config import SolverConfig
from uot.data.dataset_loader import load_image_as_color_grid

#TODO: Fix jax/non-jax


def load_config_info(config: dict)-> tuple[int, int, str, int]:
    """Get miscellaneous info from the config"""
    return (
        config['bin-number'], 
        config['batch-size'],
        config['images-dir'],
        config['pair-number']
    )

def im2mat(img):
    """Converts and image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)

def read_image(image_path: str):
    image = plt.imread(image_path)
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0
    elif image.dtype in [np.float32, np.float64]:
        image = np.clip(image, 0, 1)
    return image

def export_images(image_pairs: list, actual_images: dict, results: dict, save_path: str, solvers: list):
    """
    Export transformed images based on results keyed by (source,target) -> solver -> param_key.

    Args:
        image_pairs: list of ((source_name, source_image), (target_name, target_image))
        actual_images: dict of actual_images[image_name]: image_matrix
        results: dict of results[(source_name, target_name)][solver_name][param_key] = transformed_image
        save_path: directory to save images
        solvers: list of solver configs (to get solver.name for ordering)
    """
    os.makedirs(save_path, exist_ok=True)

    #TODO: Improve representation

    for (source_name, _), (target_name, _) in image_pairs:
        key = (source_name, target_name)
        if key not in results:
            continue

        solver_names = [solver.name for solver in solvers if solver.name in results[key]]

        param_keys_per_solver = {
            solver_name: sorted(results[key][solver_name].keys(), key=lambda pk: sorted(pk))
            for solver_name in solver_names
        }

        max_param_count = max(len(params) for params in param_keys_per_solver.values()) if param_keys_per_solver else 0

        n_rows = len(solver_names) + 2
        n_cols = max_param_count + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.expand_dims(axes, 0)
        elif n_cols == 1:
            axes = np.expand_dims(axes, 1)

        axes[0, 0].text(0.5, 0.5, source_name, ha='center', va='center', fontsize=12)
        axes[0, 0].axis("off")

        for col in range(1, n_cols):
            ax = axes[0, col]
            ax.imshow(actual_images[source_name])
            ax.axis("off")
            if col == 0:
                ax.set_title("Source Image")

        axes[-1, 0].text(0.5, 0.5, target_name, ha='center', va='center', fontsize=12)
        axes[-1, 0].axis("off")

        for col in range(1, n_cols):
            ax = axes[-1, col]
            ax.imshow(actual_images[target_name])
            ax.axis("off")
            if col == 0:
                ax.set_title("Target Image")

        for row_idx, solver_name in enumerate(solver_names, start=1):
            params = param_keys_per_solver[solver_name]

            axes[row_idx, 0].text(0.5, 0.5, solver_name, ha='center', va='center', fontsize=12)
            axes[row_idx, 0].axis("off")

            for col_idx in range(1, n_cols):
                ax = axes[row_idx, col_idx]
                if col_idx - 1 < len(params):
                    param_key = params[col_idx - 1]
                    img = results[key][solver_name][param_key]
                    shape = actual_images[source_name].shape
                    ax.imshow(img.reshape(shape))

                    if row_idx == 1:
                        param_str = ', '.join(f"{k}={v}" for k, v in sorted(param_key))
                        ax.set_title(param_str, fontsize=8)
                else:
                    ax.imshow(np.zeros_like(actual_images[source_name]))
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{source_name}-{target_name}.png"))
        plt.close(fig)


def transport_image(
    P: ArrayLike,
    source_image: ArrayLike,
    source_palette: ArrayLike,
    target_palette: ArrayLike,
    batch_size: int = 1024
) -> ArrayLike:

    P_normalized = P / np.sum(P, axis=1, keepdims=True)
    P_normalized = np.nan_to_num(P_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    projected_palette = P_normalized @ target_palette

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


def perform_experiments(batch_size: int, actual_images: dict[str, np.ndarray], image_pairs: list[tuple], solver_configs: list[SolverConfig], jax_pairs: list[tuple]|None, jax_actual_images: list[tuple]|None)-> dict:
    """Carry out the experiments specified in the config"""

    results = {}

    total_num = len(image_pairs) * sum(len(solver.param_grid) for solver in solver_configs)

    with tqdm(total=total_num, desc="Running color transfer") as pbar:

        for solver in solver_configs:

            data = image_pairs if not solver.is_jit else jax_pairs
            params = solver.param_grid

            for param_kwargs in params:

                pbar.set_description(f'Running color transfer: Solver "{solver.name}" with parameters "{param_kwargs}"')

                for source, target in data:
                    source_name, source = source
                    target_name, target = target

                    source_points, target_points = source.to_discrete()[0], target.to_discrete()[0]

                    cost = cost_euclid(source_points, target_points, use_jax=solver.is_jit)

                    key = (source_name, target_name)

                    if key not in results:
                        results[key] = {}

                    if solver.name not in results[key]:
                        results[key][solver.name] = {}

                    res = solver.solver().solve([source, target], [cost], **param_kwargs)

                    P = res['transport_plan']

                    img = im2mat(jax_actual_images[source_name]) if solver.is_jit else im2mat(actual_images[source_name])

                    param_key = frozenset(param_kwargs.items())
                    results[key][solver.name][param_key] = transport_image(P, img, source.to_discrete()[0], target.to_discrete()[0], batch_size)

                    pbar.update(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_folder = os.path.join(script_dir, "results")

    export_images(image_pairs, actual_images, results, export_folder, solver_configs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run color transfer via optimal transport")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a configuration file with experiment parameters."
    )

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    bin_num, batch_size, images_dir, pair_num = load_config_info(config)

    solver_configs = load_solvers(config=config)
    jax_needed = any(solver.is_jit for solver in solver_configs)

    images = [(image_file, load_image_as_color_grid(os.path.join(images_dir, image_file), bins_per_channel=bin_num))
           for image_file in os.listdir(images_dir)]

    actual_images = {image_file: read_image(os.path.join(images_dir, image_file)) for image_file, _ in images}

    image_pairs = list(itertools.permutations(images, 2))[:pair_num]

    jax_pairs, jax_actual_images = None, None

    if jax_needed:
        jax_images = {
            fname: load_image_as_color_grid(os.path.join(images_dir, fname), bins_per_channel=bin_num, use_jax=True)
            for fname in os.listdir(images_dir)
        }

        jax_pairs = [((source, jax_images[source]), (target, jax_images[target]))
                    for (source, _), (target, _) in image_pairs]
    
        jax_actual_images = {
            fname: jnp.array(im) for fname, im in actual_images.items()
        }

    perform_experiments(batch_size, actual_images, image_pairs, solver_configs, jax_pairs, jax_actual_images)
