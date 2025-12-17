import ot
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab
from skimage.filters import sobel
import cv2

from uot.solvers.sinkhorn import sinkhorn_divergence_with_solver

from uot.data.measure import GridMeasure, DiscreteMeasure

SINKHORN_DIVERGENCE_REG = 5e-4
SINKHORN_DIVERGENCE_MAXITER = 50000
SINKHORN_DIVERGENCE_TOL = 1e-5
SINKHORN_DIVERGENCE_BATCHSIZE = 8192
SINKHORN_DIVERGENCE_MAX_POINTS = 2000

def _prepare_sparse_measure(measure):
    if isinstance(measure, GridMeasure):
        points, weights = measure.to_discrete(include_zeros=False)
    else:
        points, weights = measure.to_discrete()

    points_np = np.asarray(points)
    weights_np = np.asarray(weights, dtype=np.float64)

    if points_np.shape[0] > SINKHORN_DIVERGENCE_MAX_POINTS:
        rng = np.random.default_rng(0)
        probs = weights_np / np.sum(weights_np)
        idx = rng.choice(
            points_np.shape[0],
            size=SINKHORN_DIVERGENCE_MAX_POINTS,
            replace=False,
            p=probs,
        )
        points_np = points_np[idx]
        weights_np = weights_np[idx]
        weights_np = weights_np / np.sum(weights_np)

    return DiscreteMeasure(points=points_np, weights=weights_np, name=getattr(measure, "name", ""))


def compute_sinhorn_divergence(source_grid: GridMeasure, target_grid: GridMeasure, batch_size: int = 1000, epsilon: float = 0.001) -> float:
    source_sparse = _prepare_sparse_measure(source_grid)
    target_sparse = _prepare_sparse_measure(target_grid)
    S = sinkhorn_divergence_with_solver(
        source_sparse, target_sparse,
        reg=SINKHORN_DIVERGENCE_REG,
        maxiter=SINKHORN_DIVERGENCE_MAXITER,
        tol=SINKHORN_DIVERGENCE_TOL,
        batch_size=SINKHORN_DIVERGENCE_BATCHSIZE,
    )

    return S['sinkhorn_divergence_w2_like']


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

    return _safe_pearson_corr(grad1, grad2)


def _safe_pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with guard rails against zero-variance vectors."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return 0.0
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    var_x = np.dot(xm, xm)
    var_y = np.dot(ym, ym)
    if var_x <= 1e-16 or var_y <= 1e-16:
        return 0.0
    corr = np.dot(xm, ym) / np.sqrt(var_x * var_y)
    return float(np.clip(corr, -1.0, 1.0))

def compute_laplacian_variance(img: np.ndarray) -> float:
    """Compute variance of Laplacian to assess sharpness (higher = sharper)."""
    gray = rgb2gray(img)
    lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
    return lap.var()
