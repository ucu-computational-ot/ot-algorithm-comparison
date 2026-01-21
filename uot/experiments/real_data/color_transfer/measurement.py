import math
import time
import gc
from typing import Dict, Union, Tuple

import jax
from jax import lax
import numpy as np
from jax import numpy as jnp

from uot.experiments.measurement import _wait_jax_finish, _require
from uot.utils.instantiate_solver import instantiate_solver
from uot.utils.types import ArrayLike
from uot.data.dataset_loader import load_matrix_as_color_grid
import uot.experiments.real_data.color_transfer.color_transfer_metrics as ct_metrics
from uot.utils.metrics.pushforward_map_metrics import extra_grid_metrics
from uot.utils.maps import barycentric_map_from_plan, mean_conditional_variance
from uot.solvers.back_and_forth.forward_pushforward import cic_pushforward_nd
from uot.solvers.back_and_forth.pushforward import adaptive_pushforward_nd
from uot.utils.logging import logger

_SOLVER_PUSHFORWARD_REGISTRY = {
    "adaptive_pushforward_nd": adaptive_pushforward_nd,
    "cic_pushforward_nd": cic_pushforward_nd,
}


def _is_identity_alpha(alpha: float) -> bool:
    return math.isclose(float(alpha), 1.0, rel_tol=1e-9, abs_tol=1e-9)


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
        # jax.clear_caches()
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
    # _require(solution, {'transport_plan', 'u_final', 'v_final', 'cost'})

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

    transferred_grid = load_matrix_as_color_grid(
        pixels=transported_image.reshape(-1, 3),
        num_channels=3,
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
        map_array = _monge_map_index_to_physical(solution['monge_map'], axes_mu)
        mask = np.asarray(mu_nd) > 0
        map_array = _maybe_soft_extend_map(map_array, mask, use_soft_extension, axes_mu)
        map_array = _apply_displacement_interpolation(
            map_array,
            axes_mu,
            displacement_alpha,
            mask=mask,
        )
        if solver_pushforward_mu is not None:
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
        if solver_pushforward_mu is not None:
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

    axes_mu_metrics, mu_nd_metrics, map_metrics, bbox = _maybe_crop_source_metrics(
        axes_mu_metrics,
        mu_nd_metrics,
        map_metrics,
        mask,
    )
    axes_nu_metrics, nu_nd_metrics, pushforward_metrics = _maybe_crop_target_metrics(
        axes_nu_metrics,
        nu_nd_metrics,
        pushforward_metrics,
        bbox=bbox,
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
    source_image = np.clip(np.asarray(prob.source_image), 0, 1)
    target_image = np.clip(np.asarray(prob.target_image), 0, 1)

    c1 = ct_metrics.compute_colorfulness(transported_image)
    c2 = ct_metrics.compute_colorfulness(target_image)

    return {
        'ssim': ct_metrics.compute_ssim_metric(transported_image, source_image),
        # 'delta_e': ct_metrics.compute_delta_e(transported_image, target_image),
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

@jax.jit
def map_pixels_by_palette_monge(image, monge_map):
    """
    image: (H, W, 3), values in [0,1]
    monge_map: (B, B, B, 3), each bin center is mapped to a color
    returns: (H, W, 3) image after color mapping
    """
    B = monge_map.shape[0]
    H, W, _ = image.shape
    pix = jnp.clip(image.reshape(-1, 3), 0.0, 1.0)
    idx = jnp.clip(jnp.round(pix * (B - 1)), 0, B - 1).astype(jnp.int32)
    mapped = monge_map[idx[:, 0], idx[:, 1], idx[:, 2]]
    mapped = jnp.clip(mapped / B, 0.0, 1.0)
    return mapped.reshape(H, W, 3)


def _convert_to_bfloat16(*arrays):
    """Convert arrays to bfloat16."""
    return tuple(arr.astype(jnp.bfloat16) for arr in arrays)


def _grid_coordinates(axes):
    grids = jnp.meshgrid(*axes, indexing='ij')
    return jnp.stack(grids, axis=-1)


def _monge_map_index_to_physical(monge_map, axes):
    arr = jnp.asarray(monge_map)
    spatial_shape = tuple(len(ax) for ax in axes)
    d = len(spatial_shape)
    if arr.shape[:len(spatial_shape)] != spatial_shape:
        if arr.shape[0] == d:
            arr = jnp.moveaxis(arr, 0, -1)
        else:
            arr = arr.reshape(spatial_shape + (-1,))
    if arr.shape[-1] != d:
        raise ValueError(f"Expected monge_map last dimension {d}, got {arr.shape[-1]}")
    spacings = jnp.array(
        [float(ax[1] - ax[0]) if ax.shape[0] > 1 else 1.0 for ax in axes],
        dtype=arr.dtype,
    )
    origins = jnp.array([float(ax[0]) for ax in axes], dtype=arr.dtype)
    reshape = (1,) * len(spatial_shape) + (d,)
    return origins.reshape(reshape) + arr * spacings.reshape(reshape)


def _pushforward_density_via_map(source_density, map_array, axes_target):
    """Scatter the source density through a map onto the target grid."""
    target_shape = tuple(len(ax) for ax in axes_target)
    d = len(target_shape)
    coords = jnp.moveaxis(jnp.asarray(map_array), -1, 0)  # (d, *shape)
    spacings = jnp.array(
        [float(ax[1] - ax[0]) if ax.shape[0] > 1 else 1.0 for ax in axes_target],
        dtype=coords.dtype,
    )
    origins = jnp.array([float(ax[0]) for ax in axes_target], dtype=coords.dtype)
    reshape = (d,) + (1,) * source_density.ndim
    spacings = jnp.maximum(spacings, 1e-12)
    idx = (coords - origins.reshape(reshape)) / spacings.reshape(reshape)
    max_idx = jnp.array(
        [max(n - 1, 0) for n in target_shape],
        dtype=coords.dtype,
    ).reshape(reshape)
    idx = jnp.clip(idx, 0.0, max_idx)

    base = jnp.floor(idx).astype(jnp.int32)
    for ax in range(d):
        limit = max(target_shape[ax] - 2, 0)
        base = base.at[ax].set(jnp.clip(base[ax], 0, limit))
    frac = idx - base.astype(coords.dtype)

    density_flat = source_density.reshape(-1)
    base_flat = base.reshape(d, -1)
    frac_flat = frac.reshape(d, -1)

    strides = []
    p = 1
    for k in range(d-1, -1, -1):
        strides.insert(0, p)
        p *= target_shape[k]
    strides = jnp.array(strides, dtype=jnp.int32).reshape(d, 1)
    out = jnp.zeros((p,), dtype=source_density.dtype)

    def corner_body(m, out_acc):
        bits = jnp.array([(m >> k) & 1 for k in range(d)], dtype=jnp.int32).reshape(d, 1)
        corner_idx = base_flat + bits
        w = jnp.where(bits == 1, frac_flat, 1.0 - frac_flat)
        w = jnp.prod(w, axis=0)
        flat_idx = jnp.sum(corner_idx * strides, axis=0)
        return out_acc.at[flat_idx].add(density_flat * w)

    out = lax.fori_loop(0, 1 << d, corner_body, out)
    return out.reshape(target_shape)


def _build_plan_grid_map(plan, target_palette, mu_nd):
    bary_map = barycentric_map_from_plan(plan, target_palette)
    spatial_shape = mu_nd.shape
    dims = target_palette.shape[1]
    expected = int(np.prod(spatial_shape))
    mu_flat = np.asarray(mu_nd).reshape(-1)
    nz_mask = mu_flat > 0
    nz_count = int(nz_mask.sum())

    if bary_map.shape[0] not in {expected, nz_count}:
        logger.warning(
            "Plan rows %s do not match grid size (%s) or nonzero bins (%s); skipping map metrics for plan.",
            bary_map.shape[0],
            expected,
            nz_count,
        )
        return None, None

    map_flat = np.zeros((expected, dims), dtype=np.asarray(bary_map).dtype)
    if bary_map.shape[0] == expected:
        map_flat[:] = np.asarray(bary_map)
    else:
        map_flat[nz_mask] = np.asarray(bary_map)
    map_array = map_flat.reshape(spatial_shape + (dims,))
    mask = nz_mask.reshape(spatial_shape)
    return jnp.asarray(map_array), mask


def _map_pixels_from_grid_map(image, grid_map):
    grid = jnp.asarray(grid_map)
    B = grid.shape[0]
    H, W, _ = image.shape
    pix = jnp.clip(image.reshape(-1, 3), 0.0, 1.0)
    idx = jnp.clip(jnp.round(pix * (B - 1)), 0, B - 1).astype(jnp.int32)
    mapped = grid[idx[:, 0], idx[:, 1], idx[:, 2]]
    return jnp.clip(mapped, 0.0, 1.0).reshape(H, W, 3)


def _map_pixels_via_palette_nearest(pixels, source_palette, target_palette, batch_size=16384):
    src = jnp.asarray(source_palette)
    tgt = jnp.asarray(target_palette)
    pix = jnp.asarray(pixels)
    N = pix.shape[0]
    outputs = []
    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        chunk = pix[start:stop]
        dists = jnp.sum((src[None, :, :] - chunk[:, None, :]) ** 2, axis=2)
        idx = jnp.argmin(dists, axis=1)
        outputs.append(tgt[idx])
    return jnp.concatenate(outputs, axis=0)


def _maybe_soft_extend_map(map_array, mask, enabled, axes=None):
    if not enabled or map_array is None or mask is None:
        return map_array
    return _soft_extend_map(map_array, mask, axes)


def _soft_extend_map(map_array, mask, axes, sigma: float | None = None):
    if axes is None:
        raise ValueError("Axes are required for soft extension.")
    arr = np.asarray(map_array, dtype=np.float64)
    mask_np = np.asarray(mask, dtype=bool)
    if arr.shape[:mask_np.ndim] != mask_np.shape:
        raise ValueError("Mask shape must match map_array spatial dimensions")

    axes_np = [np.asarray(ax, dtype=np.float64) for ax in axes]
    coords = np.stack(np.meshgrid(*axes_np, indexing='ij'), axis=-1)
    coords_flat = coords.reshape(-1, coords.shape[-1])

    values = arr.reshape(-1, arr.shape[-1])
    mask_flat = mask_np.reshape(-1)
    known_coords = coords_flat[mask_flat]
    known_values = values[mask_flat]
    missing_coords = coords_flat[~mask_flat]

    if known_coords.size == 0 or missing_coords.size == 0:
        return jnp.asarray(arr)

    if sigma is None:
        axis_scales = []
        for ax in axes_np:
            if ax.shape[0] > 1:
                axis_scales.append((ax[-1] - ax[0]) / (ax.shape[0] - 1))
            else:
                axis_scales.append(1.0)
        sigma = np.mean(axis_scales) if axis_scales else 0.1
        sigma = max(sigma * 0.5, 1e-3)

    diff = missing_coords[:, None, :] - known_coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    logits = -dist2 / (2.0 * sigma * sigma)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weight_sums = weights.sum(axis=1, keepdims=True)
    denom = np.maximum(weight_sums, 1e-12)
    safe_weights = np.divide(
        weights,
        denom,
        out=np.zeros_like(weights),
        where=weight_sums > 0,
    )
    filled_values = safe_weights @ known_values

    filled = values.copy()
    filled[~mask_flat] = filled_values

    return jnp.asarray(filled.reshape(arr.shape))

def _compute_support_bbox(mask):
    mask_np = np.asarray(mask, dtype=bool)
    if not mask_np.any():
        return None
    coords = np.stack(np.nonzero(mask_np), axis=1)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return [(int(lo), int(hi)) for lo, hi in zip(mins, maxs)]


def _slice_axes(axes, bbox):
    return [ax[start:end] for ax, (start, end) in zip(axes, bbox)]


def _slice_array(arr, bbox):
    if arr is None:
        return None
    slices = tuple(slice(start, end) for start, end in bbox)
    return arr[slices]


def _maybe_crop_source_metrics(axes, density, map_array, mask):
    if axes is None or density is None:
        return axes, density, map_array, None
    crop_mask = mask if mask is not None else (np.asarray(density) > 0)
    bbox = _compute_support_bbox(crop_mask)
    if bbox is None:
        return axes, density, map_array, None
    return (
        _slice_axes(axes, bbox),
        _slice_array(density, bbox),
        _slice_array(map_array, bbox),
        bbox,
    )


def _maybe_crop_target_metrics(axes, density, pushforward, bbox=None):
    if axes is None or density is None:
        return axes, density, pushforward
    if bbox is None:
        mask = np.asarray(density) > 0
        if pushforward is not None:
            mask = mask | (np.asarray(pushforward) > 0)
        bbox = _compute_support_bbox(mask)
        if bbox is None:
            return axes, density, pushforward
    return (
        _slice_axes(axes, bbox),
        _slice_array(density, bbox),
        _slice_array(pushforward, bbox),
    )


def _apply_displacement_interpolation(map_array, axes, alpha, coords=None, mask=None):
    if map_array is None or _is_identity_alpha(alpha):
        return map_array
    if axes is None and coords is None:
        raise ValueError("Axes are required for displacement interpolation.")
    arr = jnp.asarray(map_array)
    coord_array = coords
    if coord_array is None:
        coord_array = _grid_coordinates(axes)
    coord_array = jnp.asarray(coord_array, dtype=arr.dtype)
    blended = (1.0 - alpha) * coord_array + alpha * arr
    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=bool)
        blended = jnp.where(mask_arr[..., None], blended, coord_array)
    return blended


def _identity_index_grid(spatial_shape, dtype):
    axes = [jnp.arange(n, dtype=dtype) for n in spatial_shape]
    grids = jnp.meshgrid(*axes, indexing='ij')
    return jnp.stack(grids, axis=-1)


def _displacement_interpolate_index_map(map_array, alpha, mask=None):
    if map_array is None or _is_identity_alpha(alpha):
        return map_array
    arr = jnp.asarray(map_array)
    coords = _identity_index_grid(arr.shape[:-1], arr.dtype)
    blended = (1.0 - alpha) * coords + alpha * arr
    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=bool)
        blended = jnp.where(mask_arr[..., None], blended, coords)
    return blended


def _apply_solver_pushforward(mu_nd, solution):
    name = solution.get('pushforward_fn_name')
    psi = solution.get('v_final')
    if name is None or psi is None:
        return None
    fn = _SOLVER_PUSHFORWARD_REGISTRY.get(name)
    if fn is None:
        return None
    psi_arr = jnp.asarray(psi).reshape(mu_nd.shape)
    pushed, _ = fn(mu_nd, psi_arr)
    return pushed


def _reconstruct_target_density(plan, nu_nd):
    plan_np = np.asarray(plan)
    weights = plan_np.sum(axis=0)
    target_shape = nu_nd.shape
    total_bins = int(np.prod(target_shape))
    nu_flat = np.asarray(nu_nd).reshape(-1)
    mask = nu_flat > 0
    positive_bins = int(mask.sum())
    if weights.shape[0] != positive_bins:
        logger.warning(
            "Cannot reshape pushforward weights of size %s into target grid (%s positive bins).",
            weights.shape[0],
            positive_bins,
        )
        return None
    scattered = np.zeros(total_bins, dtype=weights.dtype)
    scattered[mask] = weights
    scattered = scattered.reshape(target_shape)
    return jnp.asarray(scattered)
