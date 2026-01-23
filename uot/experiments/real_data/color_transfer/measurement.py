import math
import time
import gc
from typing import Dict, Union, Tuple

import numpy as np
from jax import numpy as jnp

from uot.experiments.measurement import _wait_jax_finish
from uot.utils.instantiate_solver import instantiate_solver
from uot.utils.types import ArrayLike
from uot.data.dataset_loader import load_matrix_as_color_grid
import uot.experiments.real_data.color_transfer.color_transfer_metrics as ct_metrics
from uot.utils.metrics.pushforward_map_metrics import extra_grid_metrics
from uot.utils.maps import barycentric_map_from_plan, mean_conditional_variance
from uot.experiments.real_data.color_transfer.measurement_helpers import (
    _apply_displacement_interpolation,
    _apply_map_pushforward,
    _apply_map_pushforward_index,
    _apply_solver_pushforward,
    _build_plan_grid_map,
    _displacement_interpolate_index_map,
    _grid_coordinates,
    _is_identity_alpha,
    _map_pixels_from_grid_map,
    _map_pixels_via_palette_nearest,
    _maybe_crop_metrics_union,
    _maybe_soft_extend_map,
    _monge_map_index_to_physical,
    _pushforward_density_via_map,
    _reconstruct_target_density,
    map_pixels_by_palette_monge,
)
from uot.utils.logging import logger


def _build_postprocess_modes(soft_modes, displacement_alphas):
    soft_specified = soft_modes is not None
    soft_list = [bool(m) for m in soft_modes] if soft_modes else [False]
    base_soft = False
    if soft_list:
        base_soft = False if False in soft_list else soft_list[0]
    alpha_list = [float(a) for a in displacement_alphas] if displacement_alphas else [1.0]
    base_alpha = next(
        (a for a in alpha_list if _is_identity_alpha(a)),
        alpha_list[0],
    )
    modes: list[tuple[bool, float]] = []
    modes.append((base_soft, base_alpha))
    if soft_specified:
        for soft in soft_list:
            if soft == base_soft:
                continue
            modes.append((soft, base_alpha))
    for alpha in alpha_list:
        if math.isclose(alpha, base_alpha, rel_tol=1e-9, abs_tol=1e-9):
            continue
        modes.append((base_soft, alpha))
    return soft_specified, modes


def measure_color_transfer_metrics(
    prob,
    solver,
    marginals: Tuple[ArrayLike, ArrayLike],
    costs: Tuple[ArrayLike, ...],
    soft_extension_modes: list[bool] | None = None,
    displacement_alphas: list[float] | None = None,
    **kwargs
) -> list[Dict[str, Union[float, np.ndarray]]]:
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
    base_metrics = _compute_solution_metrics(solver, marginals, costs, **kwargs)
    solution = base_metrics['solution']
    axes_mu, mu_nd = marginals[0].for_grid_solver(backend="jax", dtype=jnp.float64)
    axes_nu, nu_nd = marginals[1].for_grid_solver(backend="jax", dtype=jnp.float64)
    target_palette, _ = marginals[1].to_discrete()

    plan_grid_map = None
    plan_mask = None
    if 'transport_plan' in solution:
        plan_grid_map, plan_mask = _build_plan_grid_map(
            solution['transport_plan'],
            target_palette,
            mu_nd,
        )

    soft_extension_specified, postprocess_modes = _build_postprocess_modes(
        soft_extension_modes,
        displacement_alphas,
    )

    results: list[Dict[str, Union[float, np.ndarray]]] = []
    base_without_solution = {
        key: value for key, value in base_metrics.items() if key != 'solution'
    }

    for use_soft_extension, alpha_value in postprocess_modes:
        active_soft = use_soft_extension if soft_extension_specified else False
        logger.info(f"Transporting image with soft_extension={active_soft}, displacement_alpha={alpha_value}...")
        transported_image = _process_transported_image(
            prob,
            marginals,
            solution,
            axes_mu=axes_mu,
            mu_nd=mu_nd,
            axes_nu=axes_nu,
            nu_nd=nu_nd,
            target_palette=target_palette,
            use_soft_extension=active_soft,
            displacement_alpha=alpha_value,
            plan_grid_map=plan_grid_map,
            plan_mask=plan_mask,
        )
        entry = dict(base_without_solution)
        logger.info(f"Computing metrics for transported image...")
        entry.update(_compute_distribution_metrics(transported_image, marginals[1]))
        logger.info(f"Computing map quality metrics...")
        entry.update(_compute_map_quality_metrics(
            marginals,
            solution,
            axes_mu=axes_mu,
            mu_nd=mu_nd,
            axes_nu=axes_nu,
            nu_nd=nu_nd,
            target_palette=target_palette,
            plan_grid_map=plan_grid_map,
            plan_mask=plan_mask,
            use_soft_extension=active_soft,
            displacement_alpha=alpha_value,
        ))
        logger.info(f"Computing image quality metrics...")
        entry.update(_compute_image_quality_metrics(transported_image, prob))
        if soft_extension_specified:
            entry['soft_extension'] = bool(active_soft)
        entry['displacement_alpha'] = alpha_value
        entry['transported_image'] = transported_image
        combined = dict(entry)
        combined.update(solution)
        results.append(combined)
        gc.collect()
        logger.info(f"Completed metrics for soft_extension={active_soft}, displacement_alpha={alpha_value}.")

    return results


def _compute_solution_metrics(solver, marginals, costs, **kwargs) -> Dict:
    """Compute solution and basic metrics."""
    solver_init_kwargs = kwargs or {}
    instance = instantiate_solver(solver_cls=solver, init_kwargs=solver_init_kwargs)

    start_time = time.perf_counter()
    solution = instance.solve(marginals=marginals, costs=costs, **kwargs)
    _wait_jax_finish(solution)

    return {
        "time": (time.perf_counter() - start_time),
        "cost": solution['cost'],
        "solution": solution
    }


def _process_transported_image(
    prob,
    marginals,
    solution,
    *,
    axes_mu=None,
    mu_nd=None,
    axes_nu=None,
    nu_nd=None,
    target_palette=None,
    use_soft_extension: bool = False,
    displacement_alpha: float = 1.0,
    plan_grid_map=None,
    plan_mask=None,
) -> np.ndarray:
    """Compute and process transported image."""
    transported_image = compute_transported_image(
        prob,
        marginals,
        solution,
        axes_mu=axes_mu,
        mu_nd=mu_nd,
        axes_nu=axes_nu,
        nu_nd=nu_nd,
        target_palette=target_palette,
        use_soft_extension=use_soft_extension,
        displacement_alpha=displacement_alpha,
        plan_grid_map=plan_grid_map,
        plan_mask=plan_mask,
    )
    _wait_jax_finish(transported_image)
    return jnp.clip(jnp.asarray(transported_image), 0, 1)


def _compute_distribution_metrics(transported_image: np.ndarray, target_measure) -> Dict:
    """Compute distribution-based metrics."""
    bins_per_channel = target_measure.axes[0].shape[0]
    num_channels = len(target_measure.axes)

    transferred_grid = load_matrix_as_color_grid(
        pixels=transported_image.reshape(-1, num_channels),
        num_channels=num_channels,
        bins_per_channel=bins_per_channel,
        use_jax=False
    )

    return {
        'bins_per_channel': bins_per_channel,
        'sinkhorn_divergence': ct_metrics.compute_sinhorn_divergence(
            transferred_grid, target_measure
        ),
        'kl_divergence': ct_metrics.compute_kl_divergence(
            transferred_grid, target_measure
        )
    }


def _compute_map_quality_metrics(
    marginals: Tuple[ArrayLike, ArrayLike],
    solution: Dict,
    *,
    axes_mu=None,
    mu_nd=None,
    axes_nu=None,
    nu_nd=None,
    target_palette=None,
    plan_grid_map=None,
    plan_mask=None,
    use_soft_extension: bool = False,
    displacement_alpha: float = 1.0,
) -> Dict[str, float]:
    """Compute map-quality metrics (TV, MA residual, diffuseness)."""
    if len(marginals) < 2:
        return {}

    mu_measure = marginals[0]
    nu_measure = marginals[1]
    if axes_mu is None or mu_nd is None:
        axes_mu, mu_nd = mu_measure.for_grid_solver(backend="jax", dtype=jnp.float64)
    if axes_nu is None or nu_nd is None:
        axes_nu, nu_nd = nu_measure.for_grid_solver(backend="jax", dtype=jnp.float64)
    if target_palette is None:
        target_palette, _ = nu_measure.to_discrete()

    map_array = None
    pushforward_mu = None
    diffuseness = None
    solver_pushforward_mu = _apply_solver_pushforward(mu_nd, solution)

    mask = None

    if 'monge_map' in solution:
        map_index = solution['monge_map']
        mask = np.asarray(mu_nd) > 0
        if use_soft_extension:
            # Soft extension needs physical coordinates; displacement commutes with
            # index->physical conversion, but soft extension does not.
            map_array = _monge_map_index_to_physical(map_index, axes_mu)
            map_array = _maybe_soft_extend_map(map_array, mask, True, axes_mu)
            map_array = _apply_displacement_interpolation(
                map_array,
                axes_mu,
                displacement_alpha,
                mask=mask,
            )
        else:
            map_index = _displacement_interpolate_index_map(
                map_index,
                displacement_alpha,
                mask,
            )
            map_array = _monge_map_index_to_physical(map_index, axes_mu)
        use_postprocess = use_soft_extension or not _is_identity_alpha(displacement_alpha)
        if use_postprocess:
            if use_soft_extension:
                map_pushforward_mu = _apply_map_pushforward(mu_nd, map_array, axes_mu, solution)
            else:
                map_pushforward_mu = _apply_map_pushforward_index(mu_nd, map_index, solution)
        else:
            map_pushforward_mu = None
        use_solver_pushforward = (
            solver_pushforward_mu is not None
            and not use_postprocess
        )
        if map_pushforward_mu is not None:
            pushforward_mu = map_pushforward_mu
        elif use_solver_pushforward:
            pushforward_mu = solver_pushforward_mu
        else:
            pushforward_mu = _pushforward_density_via_map(mu_nd, map_array, axes_nu)
        diffuseness = jnp.asarray(0.0, dtype=mu_nd.dtype)
    elif 'transport_plan' in solution:
        plan = jnp.asarray(solution['transport_plan'])
        if plan_grid_map is None or plan_mask is None:
            plan_grid_map, plan_mask = _build_plan_grid_map(plan, target_palette, mu_nd)
        map_array = plan_grid_map
        mask = plan_mask
        map_array = _maybe_soft_extend_map(map_array, mask, use_soft_extension, axes_mu)
        map_array = _apply_displacement_interpolation(
            map_array,
            axes_mu,
            displacement_alpha,
            mask=mask,
        )
        use_solver_pushforward = (
            solver_pushforward_mu is not None
            and not use_soft_extension
            and _is_identity_alpha(displacement_alpha)
        )
        if use_solver_pushforward:
            pushforward_mu = solver_pushforward_mu
        else:
            pushforward_mu = _reconstruct_target_density(plan, nu_nd)
        diffuseness = mean_conditional_variance(plan, target_palette)
    else:
        return {}

    axes_mu_metrics = axes_mu
    mu_nd_metrics = mu_nd
    map_metrics = map_array
    axes_nu_metrics = axes_nu
    nu_nd_metrics = nu_nd
    pushforward_metrics = pushforward_mu

    (
        axes_mu_metrics,
        mu_nd_metrics,
        map_metrics,
        axes_nu_metrics,
        nu_nd_metrics,
        pushforward_metrics,
    ) = _maybe_crop_metrics_union(
        axes_mu_metrics,
        mu_nd_metrics,
        map_metrics,
        axes_nu_metrics,
        nu_nd_metrics,
        pushforward_metrics,
        mask,
    )

    metrics = {}
    if map_metrics is not None and pushforward_metrics is not None:
        X_metrics = _grid_coordinates(axes_mu_metrics)
        grid_metrics = extra_grid_metrics(
            mu_nd=mu_nd_metrics,
            nu_nd=nu_nd_metrics,
            axes_mu=axes_mu_metrics,
            X=X_metrics,
            T=map_metrics,
            pushforward_mu=pushforward_metrics,
        )
        metrics.update({
            'tv_mu_to_nu': float(grid_metrics['tv_mu_to_nu']),
            'ma_residual_L1': float(grid_metrics['ma_residual_L1']),
            'ma_residual_Linf': float(grid_metrics['ma_residual_Linf']),
        })

    if diffuseness is not None:
        metrics['map_diffuseness'] = float(diffuseness)

    return metrics


def _compute_image_quality_metrics(
    transported_image: np.ndarray,
    prob
) -> Dict[str, float]:
    """Compute image quality and spatial metrics."""
    if hasattr(prob, "to_rgb_image"):
        transported_rgb = prob.to_rgb_image(transported_image)
        source_image = prob.source_rgb
        target_image = prob.target_rgb
    else:
        transported_rgb = transported_image
        source_image = prob.source_image
        target_image = prob.target_image
    source_image = np.clip(np.asarray(source_image), 0, 1)
    target_image = np.clip(np.asarray(target_image), 0, 1)

    c1 = ct_metrics.compute_colorfulness(transported_rgb)
    c2 = ct_metrics.compute_colorfulness(target_image)

    return {
        'ssim': ct_metrics.compute_ssim_metric(transported_rgb, source_image),
        # 'delta_e': ct_metrics.compute_delta_e(transported_image, target_image),
        'colorfulness_diff': abs(c1 - c2),
        'gradient_correlation': ct_metrics.compute_gradient_magnitude_correlation(
            transported_rgb, source_image
        ),
        'laplacian_sharpness_diff': (
            ct_metrics.compute_laplacian_variance(transported_rgb) -
            ct_metrics.compute_laplacian_variance(source_image)
        )
    }


def compute_transported_image(
    prob,
    marginals: Tuple[ArrayLike, ArrayLike],
    solution: Dict,
    *,
    axes_mu=None,
    mu_nd=None,
    axes_nu=None,
    nu_nd=None,
    target_palette=None,
    use_soft_extension: bool = False,
    displacement_alpha: float = 1.0,
    plan_grid_map=None,
    plan_mask=None,
) -> jnp.ndarray:
    """
    Compute transported image using either transport plan or Monge map.
    """
    source_palette = marginals[0].to_discrete()[0]
    if target_palette is None:
        target_palette, _ = marginals[1].to_discrete()

    if 'transport_plan' in solution:
        return _transport_image_plan(
            plan=solution['transport_plan'],
            image=prob.source_image,
            source_palette=source_palette,
            target_palette=target_palette,
            mu_measure=marginals[0],
            mu_nd=mu_nd,
            axes_mu=axes_mu,
            use_soft_extension=use_soft_extension,
             displacement_alpha=displacement_alpha,
            plan_grid_map=plan_grid_map,
            plan_mask=plan_mask,
        )
    elif 'monge_map' in solution:
        monge_map = solution['monge_map']
        needs_mask = use_soft_extension or (not _is_identity_alpha(displacement_alpha))
        mask = None
        if needs_mask:
            if mu_nd is None or axes_mu is None:
                axes_mu, mu_nd = marginals[0].for_grid_solver(backend="jax", dtype=jnp.float64)
            mask = np.asarray(mu_nd) > 0
        if use_soft_extension:
            monge_map = _maybe_soft_extend_map(monge_map, mask, True, axes_mu)
        monge_map = _displacement_interpolate_index_map(monge_map, displacement_alpha, mask)
        return map_pixels_by_palette_monge(prob.source_image, monge_map)
    else:
        raise ValueError("Solution must contain either 'transport_plan' or 'monge_map'")


def _transport_image_plan(
    plan: jnp.ndarray,
    image: jnp.ndarray,
    source_palette: jnp.ndarray,
    target_palette: jnp.ndarray,
    mu_measure=None,
    mu_nd=None,
    axes_mu=None,
    use_soft_extension: bool = False,
    displacement_alpha: float = 1.0,
    plan_grid_map=None,
    plan_mask=None,
) -> jnp.ndarray:
    """Transform image using transport plan."""
    needs_grid = use_soft_extension or (not _is_identity_alpha(displacement_alpha))
    grid_map = plan_grid_map
    mask = plan_mask
    if needs_grid:
        if (grid_map is None) or (mask is None):
            if mu_nd is None or axes_mu is None:
                axes_mu, mu_nd = mu_measure.for_grid_solver(backend="jax", dtype=jnp.float64)
            grid_map, mask = _build_plan_grid_map(plan, target_palette, mu_nd)
        if grid_map is not None and mask is not None and axes_mu is not None:
            working_map = grid_map
            if use_soft_extension:
                working_map = _maybe_soft_extend_map(working_map, mask, True, axes_mu)
            working_map = _apply_displacement_interpolation(
                working_map,
                axes_mu,
                displacement_alpha,
                mask=mask,
            )
            return _map_pixels_from_grid_map(image, working_map)
        elif use_soft_extension:
            logger.warning("Soft extension requested but plan grid map could not be built; using direct barycentric mapping.")

    T_centers = barycentric_map_from_plan(plan, target_palette)
    if not _is_identity_alpha(displacement_alpha):
        T_centers = (1.0 - displacement_alpha) * jnp.asarray(source_palette) + displacement_alpha * T_centers
    H, W, C = image.shape
    pixels = image.reshape(-1, C)
    mapped = _map_pixels_via_palette_nearest(
        pixels,
        source_palette,
        T_centers,
    )
    return mapped.reshape(H, W, C).astype(jnp.float32)
