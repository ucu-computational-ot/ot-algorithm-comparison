import time
from typing import Dict, Union, Tuple

import jax
import numpy as np
from jax import numpy as jnp

from uot.experiments.measurement import _wait_jax_finish, _require
from uot.utils.instantiate_solver import instantiate_solver
from uot.utils.types import ArrayLike
from uot.data.dataset_loader import load_matrix_as_color_grid
import uot.experiments.real_data.color_transfer.color_transfer_metrics as ct_metrics


def measure_color_transfer_metrics(
    prob,
    solver,
    marginals: Tuple[ArrayLike, ArrayLike],
    costs: Tuple[ArrayLike, ...],
    **kwargs
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Measure color transfer metrics for a given problem and solver.
    
    Args:
        prob: Color transfer problem instance
        solver: Solver class to use
        marginals: Tuple of source and target measures
        costs: Tuple of cost matrices
        **kwargs: Additional arguments for solver
        
    Returns:
        Dictionary containing computed metrics and transported image
    """
    metrics = _compute_solution_metrics(solver, marginals, costs, **kwargs)
    # крч тут ебаная хуйня с типами оно транслирует это в память оперативки используя нумпай
    transported_image = _process_transported_image(prob, marginals, metrics['solution'])
    metrics.update(_compute_distribution_metrics(transported_image, marginals[1]))
    metrics.update(_compute_image_quality_metrics(transported_image, prob))
    metrics['transported_image'] = transported_image
    metrics.pop('solution', None)
    return metrics


def _compute_solution_metrics(solver, marginals, costs, **kwargs) -> Dict:
    """Compute solution and basic metrics."""
    solver_init_kwargs = kwargs or {}
    instance = instantiate_solver(solver_cls=solver, init_kwargs=solver_init_kwargs)
    
    start_time = time.perf_counter()
    solution = instance.solve(marginals=marginals, costs=costs, **kwargs)
    _wait_jax_finish(solution)
    # _require(solution, {'transport_plan', 'u_final', 'v_final', 'cost'})
    
    return {
        "time": (time.perf_counter() - start_time),
        "cost": solution['cost'],
        "solution": solution
    }


def _process_transported_image(prob, marginals, solution) -> np.ndarray:
    """Compute and process transported image."""
    transported_image = compute_transported_image(prob, marginals, solution)
    _wait_jax_finish(transported_image)
    # а вот какого хуя тут нумпай а не джакс?
    return jnp.clip(jnp.asarray(transported_image), 0, 1)


def _compute_distribution_metrics(transported_image: np.ndarray, target_measure) -> Dict:
    """Compute distribution-based metrics."""
    transferred_grid = load_matrix_as_color_grid(
        pixels=transported_image.reshape(-1, 3),
        num_channels=3,
        bins_per_channel=32,
        use_jax=True
    )
    # ).get_jax() # так так так, ебем в джакс?
    
    return {
        'wasserstein_distance': ct_metrics.compute_wasserstein_distance(
            transferred_grid, target_measure
        ),
        'kl_divergence': ct_metrics.compute_kl_divergence(
            transferred_grid, target_measure
        )
    }


def _compute_image_quality_metrics(
    transported_image: np.ndarray,
    prob
) -> Dict[str, float]:
    """Compute image quality and spatial metrics."""
    source_image = np.clip(np.asarray(prob.source_image), 0, 1)
    target_image = np.clip(np.asarray(prob.target_image), 0, 1)
    
    c1 = ct_metrics.compute_colorfulness(transported_image)
    c2 = ct_metrics.compute_colorfulness(target_image)
    
    return {
        'ssim': ct_metrics.compute_ssim_metric(transported_image, source_image),
        'delta_e': ct_metrics.compute_delta_e(transported_image, target_image),
        'colorfulness_diff': abs(c1 - c2),
        'gradient_correlation': ct_metrics.compute_gradient_magnitude_correlation(
            transported_image, source_image
        ),
        'laplacian_sharpness_diff': (
            ct_metrics.compute_laplacian_variance(transported_image) -
            ct_metrics.compute_laplacian_variance(source_image)
        )
    }


def compute_transported_image(
    prob,
    marginals: Tuple[ArrayLike, ArrayLike],
    solution: Dict
) -> jnp.ndarray:
    """
    Compute transported image using either transport plan or Monge map.
    """
    source_palette, target_palette = (
        marginals[0].to_discrete()[0],
        marginals[1].to_discrete()[0]
    )
    
    if 'transport_plan' in solution:
        return _transport_image_plan(
            plan=solution['transport_plan'],
            image=prob.source_image,
            source_palette=source_palette,
            target_palette=target_palette
        )
    elif 'monge_map' in solution:
        return _transport_image_monge(
            monge_map=solution['monge_map'],
            image=prob.source_image,
            source_palette=source_palette,
            target_palette=target_palette
        )
    else:
        raise ValueError("Solution must contain either 'transport_plan' or 'monge_map'")


@jax.jit
def _transport_image_plan(
    plan: jnp.ndarray,
    image: jnp.ndarray,
    source_palette: jnp.ndarray,
    target_palette: jnp.ndarray,
) -> jnp.ndarray:
    """Transform image using transport plan."""
    plan, image, source_palette, target_palette = _convert_to_bfloat16(
        plan, image, source_palette, target_palette
    )
    T_centers = _compute_mapped_centers(plan, target_palette)
    H, W, C = image.shape
    pixels = image.reshape(-1, C)
    
    @jax.vmap
    def transform_pixel(pixel):
        dists = jnp.sum((source_palette - pixel[None, :])**2, axis=1)
        idx = jnp.argmin(dists)
        return T_centers[idx]
    
    return transform_pixel(pixels).reshape(H, W, C).astype(jnp.float32)


@jax.jit
def _transport_image_monge(
    monge_map: jnp.ndarray,
    image: jnp.ndarray,
    source_palette: jnp.ndarray,
    target_palette: jnp.ndarray,
) -> jnp.ndarray:
    """Transform image using Monge map."""
    image, source_palette, target_palette = _convert_to_bfloat16(
        image, source_palette, target_palette
    )
    H, W, C = image.shape
    pixels = image.reshape(-1, C)
    
    @jax.vmap
    def transform_pixel(pixel):
        dists = jnp.sum((source_palette - pixel[None, :])**2, axis=1)
        source_idx = jnp.argmin(dists)
        target_idx = monge_map[source_idx]
        return target_palette[target_idx]
    
    return transform_pixel(pixels).reshape(H, W, C).astype(jnp.float32)


def _convert_to_bfloat16(*arrays):
    """Convert arrays to bfloat16."""
    return tuple(arr.astype(jnp.bfloat16) for arr in arrays)


def _compute_mapped_centers(plan: jnp.ndarray, target_palette: jnp.ndarray) -> jnp.ndarray:
    """Compute mapped palette centers."""
    return jnp.nan_to_num(
        (plan @ target_palette) / plan.sum(axis=1, keepdims=True),
        copy=False
    )