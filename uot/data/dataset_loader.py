from uot.data.measure import DiscreteMeasure, GridMeasure
from PIL import Image
import numpy as np
import jax.numpy as jnp
import os

def load_csv_as_discrete(path: str) -> DiscreteMeasure:
    "Loads the discrete measure from the defined path to the data."
    raise NotImplementedError()


def load_image_as_color_grid(
    path: str,
    bins_per_channel: int = 32,
    use_jax: bool = False
) -> GridMeasure:
    """
    Loads the image at `path` and converts it to a color grid measure.
    Supports either NumPy or JAX backends depending on `use_jax`.
    """
    lib = jnp if use_jax else np

    image = Image.open(path)
    data = lib.asarray(np.asarray(image))

    if data.ndim == 2:
        data = data[:, :, None]

    num_channels = data.shape[2]
    pixels = data.reshape(-1, num_channels).astype(lib.float32) / 255.0

    bin_edges = lib.linspace(0.0, 1.0, bins_per_channel + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_edges_np = np.linspace(0.0, 1.0, bins_per_channel + 1)
    bins = [np.clip(np.digitize(np.asarray(pixels[:, ch]), bin_edges_np) - 1, 0, bins_per_channel - 1)
            for ch in range(num_channels)]

    grid_shape = tuple([bins_per_channel] * num_channels)
    weights_nd = lib.zeros(grid_shape, dtype=lib.float32)

    for idx in zip(*bins):
        weights_nd = weights_nd.at[idx].add(1) if use_jax else weights_nd.__setitem__(idx, weights_nd[idx] + 1)

    axes = [bin_centers for _ in range(num_channels)]

    image_name = os.path.basename(path)
    return GridMeasure(axes=axes, weights_nd=weights_nd, name=image_name, normalize=True)