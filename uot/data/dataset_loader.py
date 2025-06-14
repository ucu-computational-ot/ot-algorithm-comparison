from uot.data.measure import DiscreteMeasure, GridMeasure
from PIL import Image
import numpy as np
import os

def load_csv_as_discrete(path: str) -> DiscreteMeasure:
    "Loads the discrete measure from the defined path to the data."
    raise NotImplementedError()


def load_image_as_color_grid(path: str, bins_per_channel: int = 32) -> GridMeasure:
    "Loads the image as the color grid measure from the defined path."
    image = Image.open(path)
    data = np.asarray(image)

    if data.ndim == 2:
        data = data[:, :, None]

    num_channels = data.shape[2]
    pixels = data.reshape(-1, num_channels).astype(np.float32) / 255.0

    bin_edges = np.linspace(0.0, 1.0, bins_per_channel + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bins = [np.clip(np.digitize(pixels[:, ch], bin_edges) - 1, 0, bins_per_channel - 1)
            for ch in range(num_channels)]

    grid_shape = tuple([bins_per_channel] * num_channels)
    weights_nd = np.zeros(grid_shape, dtype=np.float32)

    for idx in zip(*bins):
        weights_nd[idx] += 1

    axes = [bin_centers for _ in range(num_channels)]

    image_name = os.path.basename(path)
    return GridMeasure(axes=axes, weights_nd=weights_nd, name=image_name, normalize=True)