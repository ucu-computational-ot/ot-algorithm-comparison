from uot.data.measure import DiscreteMeasure, GridMeasure
from PIL import Image
import numpy as np
import jax.numpy as jnp
from jax.ops import segment_sum
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
    Fast and vectorized version for both NumPy and JAX.
    """
    lib = jnp if use_jax else np

    image = Image.open(path)
    data = lib.asarray(np.asarray(image))

    if data.ndim == 2:
        data = data[:, :, None]

    num_channels = data.shape[2]
    pixels = data.reshape(-1, num_channels).astype(lib.float32) / 255.0

    bin_edges = np.linspace(0.0, 1.0, bins_per_channel + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bins = np.stack([
        np.clip(np.digitize(np.asarray(pixels[:, ch]), bin_edges) - 1, 0, bins_per_channel - 1)
        for ch in range(num_channels)
    ], axis=1)

    if use_jax:

        multipliers = np.array([bins_per_channel ** i for i in reversed(range(num_channels))])
        flat_bins = (bins @ multipliers).astype(np.int32)

        weights_1d = jnp.ones(flat_bins.shape[0], dtype=jnp.float32)
        flat_hist = segment_sum(weights_1d, flat_bins, bins_per_channel ** num_channels)
        weights_nd = flat_hist.reshape([bins_per_channel] * num_channels)
    else:
        weights_nd = np.zeros([bins_per_channel] * num_channels, dtype=np.float32)
        for idx in bins:
            weights_nd[tuple(idx)] += 1

    axes = [bin_centers for _ in range(num_channels)]

    image_name = os.path.basename(path)
    return GridMeasure(axes=axes, weights_nd=weights_nd, name=image_name, normalize=True)