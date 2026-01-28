from uot.data.measure import DiscreteMeasure, GridMeasure
from uot.utils.types import ArrayLike
from PIL import Image
import numpy as np
import jax.numpy as jnp
from jax.ops import segment_sum
from skimage.color import rgb2lab, lab2rgb


def _normalize_lab(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    l = np.clip(lab[..., 0], 0.0, 100.0) / 100.0
    a = (lab[..., 1] + 128.0) / 255.0
    b = (lab[..., 2] + 128.0) / 255.0
    return np.stack([l, a, b], axis=-1)


def _sanitize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(rgb, 0.0, 1.0)


def _denormalize_lab(norm_lab: np.ndarray) -> np.ndarray:
    norm_lab = np.asarray(norm_lab, dtype=np.float64)
    l = np.clip(norm_lab[..., 0], 0.0, 1.0) * 100.0
    a = np.clip(norm_lab[..., 1], 0.0, 1.0) * 255.0 - 128.0
    b = np.clip(norm_lab[..., 2], 0.0, 1.0) * 255.0 - 128.0
    return np.stack([l, a, b], axis=-1)


def convert_rgb_to_color_space(rgb: np.ndarray, color_space: str) -> np.ndarray:
    space = color_space.strip().lower()
    if space == "rgb":
        return _sanitize_rgb(rgb)
    if space in {"lab", "cielab"}:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lab = rgb2lab(_sanitize_rgb(rgb))
        lab = np.nan_to_num(lab, nan=0.0, posinf=0.0, neginf=0.0)
        return _normalize_lab(lab)
    raise ValueError(f"Unsupported color space: {color_space}")


def convert_color_space_to_rgb(image: np.ndarray, color_space: str) -> np.ndarray:
    space = color_space.strip().lower()
    if space == "rgb":
        return np.asarray(image, dtype=np.float64)
    if space in {"lab", "cielab"}:
        return lab2rgb(_denormalize_lab(image))
    raise ValueError(f"Unsupported color space: {color_space}")

def load_csv_as_discrete(path: str) -> DiscreteMeasure:
    "Loads the discrete measure from the defined path to the data."
    raise NotImplementedError()


def load_matrix_as_color_grid(pixels: ArrayLike, num_channels: int, bins_per_channel: int = 32, use_jax: bool = False) -> GridMeasure:
    """
    Converts a matrix to a color grid measure.
    Fast and vectorized version for both NumPy and JAX.
    """
    bin_edges = np.linspace(0.0, 1.0, bins_per_channel + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bins = np.stack([
        np.clip(np.digitize(np.asarray(pixels[:, ch]), bin_edges) - 1, 0, bins_per_channel - 1)
        for ch in range(num_channels)
    ], axis=1)

    if use_jax:

        multipliers = np.array([bins_per_channel ** i for i in reversed(range(num_channels))])
        flat_bins = (bins @ multipliers).astype(np.int32)

        weights_1d = jnp.ones(flat_bins.shape[0], dtype=jnp.float64)
        flat_hist = segment_sum(weights_1d, flat_bins, bins_per_channel ** num_channels)
        weights_nd = flat_hist.reshape([bins_per_channel] * num_channels)
    else:
        weights_nd = np.zeros([bins_per_channel] * num_channels, dtype=np.float64)
        for idx in bins:
            weights_nd[tuple(idx)] += 1

    axes = [bin_centers for _ in range(num_channels)]

    return GridMeasure(axes=axes, weights_nd=weights_nd, normalize=True)


def load_image_as_color_grid(
    path: str,
    bins_per_channel: int = 32,
    use_jax: bool = False,
    *,
    color_space: str = "rgb",
    active_channels: list[int] | None = None,
) -> GridMeasure:
    """
    Loads the image at `path` and converts it to a color grid measure.
    Fast and vectorized version for both NumPy and JAX.
    """
    lib = jnp if use_jax else np

    image = Image.open(path).convert("RGB")
    data = np.asarray(image)

    if data.ndim == 2:
        data = data[:, :, None]

    rgb = data.astype(np.float64) / 255.0
    color_data = convert_rgb_to_color_space(rgb, color_space)
    if active_channels is not None:
        color_data = color_data[..., active_channels]
    num_channels = color_data.shape[2]
    pixels = color_data.reshape(-1, num_channels).astype(lib.float64)

    return load_matrix_as_color_grid(
        pixels=pixels,
        num_channels=num_channels,
        bins_per_channel=bins_per_channel,
        use_jax=use_jax
    )
