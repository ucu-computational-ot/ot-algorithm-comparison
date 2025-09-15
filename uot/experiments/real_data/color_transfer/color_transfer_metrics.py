import ot
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab
from skimage.filters import sobel
from scipy.stats import pearsonr
import cv2

from uot.data.measure import GridMeasure


def compute_wasserstein_distance(source_grid: GridMeasure, target_grid: GridMeasure) -> float:
    src_support, src_weights = source_grid.to_discrete(include_zeros=True)
    tgt_support, tgt_weights = target_grid.to_discrete(include_zeros=True)
    # converting to numpy to avoid jax tree_map issues on v6.0
    src_weights = np.asarray(src_weights)
    tgt_weights = np.asarray(tgt_weights)
    cost = np.asarray(ot.dist(src_support, tgt_support, metric='euclidean'))
    return ot.emd2(src_weights, tgt_weights, cost)


def compute_kl_divergence(source_grid: GridMeasure, target_grid: GridMeasure) -> float:
    _, src_weights = source_grid.to_discrete(include_zeros=True)
    _, tgt_weights = target_grid.to_discrete(include_zeros=True)
    
    src_weights = src_weights / np.sum(src_weights)
    src_weights = np.clip(src_weights, 1e-12, 1)

    tgt_weights = tgt_weights / np.sum(tgt_weights)
    tgt_weights = np.clip(tgt_weights, 1e-12, 1)

    return np.sum(src_weights * np.log(src_weights /tgt_weights))

def compute_ssim_metric(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two RGB images."""
    return ssim(img1, img2, channel_axis=-1, data_range=1.0)

def compute_delta_e(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute average ΔE (CIE76) between two images."""
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    delta_e = np.linalg.norm(lab1 - lab2, axis=-1)
    return np.mean(delta_e)

def compute_colorfulness(image: np.ndarray) -> float:
    """Compute Hasler-Susstrunk colorfulness metric."""
    R, G, B = image[..., 0], image[..., 1], image[..., 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)

    return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

def rgb2gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using luminance-preserving weights."""
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

def compute_gradient_magnitude_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute correlation of gradient magnitudes between two grayscale images."""
    gray1 = rgb2gray(img1)
    gray2 = rgb2gray(img2)
    grad1 = sobel(gray1)
    grad2 = sobel(gray2)

    if np.allclose(grad1, grad1[0]) or np.allclose(grad2, grad2[0]):
        return 0.0

    return pearsonr(grad1.ravel(), grad2.ravel())[0]

def compute_laplacian_variance(img: np.ndarray) -> float:
    """Compute variance of Laplacian to assess sharpness (higher = sharper)."""
    gray = rgb2gray(img)
    lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
    return lap.var()