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

class ImageData:

    image_dir = None

    def __init__(self, name: str):
        self.name = name
        self._np_grid = load_image_as_color_grid(os.path.join(ImageData.images_dir, name))
        self._np_image = read_image(os.path.join(ImageData.images_dir, name))
        self._jax_grid = None
        self._jax_image = None
    
    def get_grid(self, use_jax: bool = False) -> ArrayLike:
        """Get the color grid of the image, either as numpy or jax array"""
        if use_jax:
            if self._jax_grid is None:
                self._jax_grid = self._np_grid.get_jax()
            return self._jax_grid
        return self._np_grid

    def get_image(self, use_jax: bool = False) -> ArrayLike:
        """Get the image as numpy or jax array"""
        if use_jax:
            if self._jax_image is None:
                self._jax_image = jnp.array(self._np_image)
            return self._jax_image
        return self._np_image

    def get_image_shape(self) -> tuple[int, int, int]:
        """Get the shape of the image"""
        return self._np_image.shape

    @classmethod
    def set_image_dir(cls, image_dir: str):
        """Set the directory where images are stored"""
        cls.images_dir = image_dir



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

def export_images(data: dict[str, ImageData], image_pairs: list[tuple], results: dict, save_path: str, solvers: list):
    """
    Export transformed images based on results keyed by (source,target) -> solver -> param_key.

    Args:
        data (dict): Dictionary of ImageData objects keyed by image names.
        image_pairs (list): List of tuples containing pairs of image names.
        results (dict): Dictionary containing the results of the experiments.
        save_path (str): Directory where the images will be saved.
        solvers (list): List of solver configurations used in the experiments.
    """
    os.makedirs(save_path, exist_ok=True)

    for key in image_pairs:
        source_name, target_name = key
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
            ax.imshow(data[source_name].get_image())
            ax.axis("off")
            if col == 0:
                ax.set_title("Source Image")

        axes[-1, 0].text(0.5, 0.5, target_name, ha='center', va='center', fontsize=12)
        axes[-1, 0].axis("off")

        for col in range(1, n_cols):
            ax = axes[-1, col]
            ax.imshow(data[target_name].get_image())
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
                    shape = data[source_name].get_image_shape()
                    ax.imshow(img.reshape(shape))

                    if row_idx == 1:
                        param_str = ', '.join(f"{k}={v}" for k, v in sorted(param_key))
                        ax.set_title(param_str, fontsize=8)
                else:
                    ax.imshow(np.zeros(data[source_name].get_image_shape()))
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


def perform_experiments(batch_size: int, data: dict[str, ImageData], image_pairs: list[tuple], solver_configs: list[SolverConfig])-> dict:
    """Carry out the experiments specified in the config"""

    results = {}

    total_num = len(image_pairs) * sum(len(solver.param_grid) for solver in solver_configs)

    with tqdm(total=total_num, desc="Running color transfer") as pbar:

        for solver in solver_configs:

            params = solver.param_grid

            for param_kwargs in params:

                pbar.set_description(f'Running color transfer: Solver "{solver.name}" with parameters "{param_kwargs}"')

                for (source_name, target_name) in image_pairs:

                    source = data[source_name]
                    target = data[target_name]

                    source_points, target_points = source.get_grid(use_jax=solver.is_jit).to_discrete()[0], target.get_grid(use_jax=solver.is_jit).to_discrete()[0]
                    cost = cost_euclid(source_points, target_points, use_jax=solver.is_jit)

                    key = (source_name, target_name)

                    if key not in results:
                        results[key] = {}
                    
                    if solver.name not in results[key]:
                        results[key][solver.name] = {}
                    
                    res = solver.solver().solve(
                                        [source.get_grid(use_jax=solver.is_jit), target.get_grid(use_jax=solver.is_jit)],
                                        [cost],
                                        **param_kwargs
                                        )

                    P = res['transport_plan']
                    img = im2mat(source.get_image(use_jax=solver.is_jit))
                    param_key = frozenset(param_kwargs.items())

                    results[key][solver.name][param_key] = transport_image(
                        P, img, source_points, target_points, batch_size
                    )

                    pbar.update(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_folder = os.path.join(script_dir, "results")

    export_images(data, image_pairs, results, export_folder, solver_configs)


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
    ImageData.set_image_dir(images_dir)

    data = {name: ImageData(name) for name in os.listdir(images_dir)}

    image_pairs = list(itertools.permutations(data, 2))[:pair_num]

    perform_experiments(batch_size, data, image_pairs, solver_configs)
