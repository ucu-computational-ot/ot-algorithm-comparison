import math

import jax
from jax import lax
import numpy as np
from jax import numpy as jnp

from uot.solvers.back_and_forth.forward_pushforward import cic_pushforward_nd
from uot.solvers.back_and_forth.pushforward import adaptive_pushforward_nd
from uot.solvers.back_and_forth.map_pushforward import (
    cic_pushforward_map_nd,
    adaptive_pushforward_map_nd,
)
from uot.solvers.back_and_forth.monge_map import map_index_to_unit
from uot.solvers.back_and_forth.solver import BackNForthSqEuclideanSolver
from uot.utils.maps import barycentric_map_from_plan
from uot.utils.logging import logger

_SOLVER_PUSHFORWARD_REGISTRY = {
    "adaptive_pushforward_nd": adaptive_pushforward_nd,
    "cic_pushforward_nd": cic_pushforward_nd,
}

_SOLVER_MAP_PUSHFORWARD_REGISTRY = {
    "adaptive_pushforward_nd": adaptive_pushforward_map_nd,
    "cic_pushforward_nd": cic_pushforward_map_nd,
}

_monge_map_index_to_physical = BackNForthSqEuclideanSolver._monge_map_index_to_physical


def _is_identity_alpha(alpha: float) -> bool:
    return math.isclose(float(alpha), 1.0, rel_tol=1e-9, abs_tol=1e-9)


@jax.jit
def map_pixels_by_palette_monge(image, monge_map):
    """
    image: (H, W, 3), values in [0,1]
    monge_map: (B, B, B, 3), each bin center is mapped to a color
    returns: (H, W, 3) image after color mapping
    """
    B = monge_map.shape[0]
    H, W, C = image.shape
    pix = jnp.clip(image.reshape(-1, C), 0.0, 1.0)
    idx = jnp.clip(jnp.round(pix * (B - 1)), 0, B - 1).astype(jnp.int32)
    indices = tuple(idx[:, i] for i in range(C))
    mapped = monge_map[indices]
    mapped = jnp.clip(mapped / B, 0.0, 1.0)
    return mapped.reshape(H, W, C)


def _grid_coordinates(axes):
    grids = jnp.meshgrid(*axes, indexing='ij')
    return jnp.stack(grids, axis=-1)


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
    H, W, C = image.shape
    pix = jnp.clip(image.reshape(-1, C), 0.0, 1.0)
    idx = jnp.clip(jnp.round(pix * (B - 1)), 0, B - 1).astype(jnp.int32)
    indices = tuple(idx[:, i] for i in range(C))
    mapped = grid[indices]
    return jnp.clip(mapped, 0.0, 1.0).reshape(H, W, C)


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


def _union_bbox(bbox_a, bbox_b):
    if bbox_a is None:
        return bbox_b
    if bbox_b is None:
        return bbox_a
    return [
        (min(a0, b0), max(a1, b1))
        for (a0, a1), (b0, b1) in zip(bbox_a, bbox_b)
    ]


def _maybe_crop_metrics_union(
    axes_mu,
    mu_nd,
    map_array,
    axes_nu,
    nu_nd,
    pushforward_mu,
    mask,
):
    if axes_mu is None or mu_nd is None or axes_nu is None or nu_nd is None:
        return axes_mu, mu_nd, map_array, axes_nu, nu_nd, pushforward_mu
    source_mask = mask if mask is not None else (np.asarray(mu_nd) > 0)
    target_mask = np.asarray(nu_nd) > 0
    if pushforward_mu is not None:
        target_mask = target_mask | (np.asarray(pushforward_mu) > 0)
    bbox_source = _compute_support_bbox(source_mask)
    bbox_target = _compute_support_bbox(target_mask)
    bbox = _union_bbox(bbox_source, bbox_target)
    if bbox is None:
        return axes_mu, mu_nd, map_array, axes_nu, nu_nd, pushforward_mu
    return (
        _slice_axes(axes_mu, bbox),
        _slice_array(mu_nd, bbox),
        _slice_array(map_array, bbox),
        _slice_axes(axes_nu, bbox),
        _slice_array(nu_nd, bbox),
        _slice_array(pushforward_mu, bbox),
    )


def _apply_solver_pushforward(mu_nd, solution):
    name = solution.get('pushforward_fn_name')
    psi = solution.get('v_final')
    if name is None or psi is None:
        return None
    fn = _SOLVER_PUSHFORWARD_REGISTRY.get(name)
    if fn is None:
        return None
    psi_arr = jnp.asarray(psi).reshape(mu_nd.shape)
    pushed, _ = fn(mu_nd, -psi_arr)
    return pushed


def _normalize_map_to_unit(map_array, axes):
    arr = jnp.asarray(map_array)
    spatial_shape = tuple(len(ax) for ax in axes)
    d = len(spatial_shape)
    if arr.shape[:len(spatial_shape)] != spatial_shape:
        if arr.shape[0] == d:
            arr = jnp.moveaxis(arr, 0, -1)
        else:
            arr = arr.reshape(spatial_shape + (d,))
    spacings = jnp.array(
        [float(ax[1] - ax[0]) if ax.shape[0] > 1 else 1.0 for ax in axes],
        dtype=arr.dtype,
    )
    origins = jnp.array([float(ax[0]) for ax in axes], dtype=arr.dtype)
    n_vec = jnp.array(spatial_shape, dtype=arr.dtype)
    reshape = (1,) * len(spatial_shape) + (d,)
    min_edge = origins - 0.5 * spacings
    max_edge = origins + spacings * (n_vec - 0.5)
    denom = jnp.maximum(max_edge - min_edge, 1e-12)
    unit = (arr - min_edge.reshape(reshape)) / denom.reshape(reshape)
    return jnp.clip(unit, 0.0, 1.0)


def _apply_map_pushforward_index(mu_nd, map_array, solution):
    name = solution.get('pushforward_fn_name')
    if name is None:
        return None
    fn = _SOLVER_MAP_PUSHFORWARD_REGISTRY.get(name)
    if fn is None:
        return None
    map_unit = map_index_to_unit(
        jnp.moveaxis(jnp.asarray(map_array), -1, 0),
        center_offset=(name == "adaptive_pushforward_nd"),
    )
    map_unit = jnp.clip(jnp.moveaxis(map_unit, 0, -1), 0.0, 1.0)
    pushed, _ = fn(mu_nd, map_unit)
    return pushed


def _apply_map_pushforward(mu_nd, map_array, axes, solution):
    name = solution.get('pushforward_fn_name')
    if name is None:
        return None
    fn = _SOLVER_MAP_PUSHFORWARD_REGISTRY.get(name)
    if fn is None:
        return None
    map_unit = _normalize_map_to_unit(map_array, axes)
    pushed, _ = fn(mu_nd, map_unit)
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
