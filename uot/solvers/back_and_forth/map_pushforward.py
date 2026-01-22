from jax import lax
from jax import numpy as jnp

from .pushforward import (
    _binary_corners,
    _multilinear_interpolate_cell_center,
    _adjacent_max,
)


def _coerce_map_array(map_array, shape):
    d = len(shape)
    arr = jnp.asarray(map_array)
    if arr.ndim == d:
        arr = arr[..., None]
    if arr.shape[0] == d and arr.ndim == d + 1:
        arr = jnp.moveaxis(arr, 0, -1)
    elif arr.shape[-1] != d:
        arr = arr.reshape(shape + (d,))
    return arr


def cic_pushforward_map_nd(density, map_array):
    """
    Pushforward a density field along a provided map (cell-centered coordinates).
    - density: (n_1, ..., n_d) array of nonnegative values
    - map_array: (n_1, ..., n_d, d) in unit coordinates on [0,1]^d
    Returns:
    - new_density: (n_1, ..., n_d) array of nonnegative values
    """
    shape = density.shape
    d = density.ndim
    n_vec = jnp.array(shape, dtype=jnp.float32)

    map_arr = _coerce_map_array(map_array, shape)
    coords = jnp.moveaxis(map_arr, -1, 0)  # (d, *shape)

    s_raw = coords * n_vec.reshape((-1,) + (1,) * d)
    s = jnp.zeros_like(s_raw)
    for ax in range(d):
        s_ax = jnp.clip(s_raw[ax], 0.0, shape[ax] - 1.0)
        s = s.at[ax].set(s_ax)

    base = jnp.floor(s).astype(jnp.int32)
    for ax in range(d):
        base_ax = jnp.clip(base[ax], 0, shape[ax] - 2)
        base = base.at[ax].set(base_ax)
    frac = s - base.astype(jnp.float32)

    density_flat = density.reshape(-1)
    base_flat = base.reshape(d, -1)
    frac_flat = frac.reshape(d, -1)

    strides_py = []
    p = 1
    for k in range(d - 1, -1, -1):
        strides_py.insert(0, p)
        p *= shape[k]
    strides = jnp.array(strides_py, dtype=jnp.int32).reshape(d, 1)

    out = jnp.zeros_like(density_flat)

    def corner_body(m, out_acc):
        bits = jnp.array([(m >> k) & 1 for k in range(d)], dtype=jnp.int32).reshape(d, 1)
        corner_idx = base_flat + bits
        w = jnp.where(bits == 1, frac_flat, 1.0 - frac_flat)
        w = jnp.prod(w, axis=0)
        flat_idx = jnp.sum(corner_idx * strides, axis=0)
        return out_acc.at[flat_idx].add(density_flat * w)

    out = lax.fori_loop(0, 1 << d, corner_body, out)
    return out.reshape(shape), None


def adaptive_pushforward_map_nd(density, map_array):
    """
    Adaptive pushforward using a provided map (cell-centered coordinates).
    - density: (n_1, ..., n_d) array of nonnegative values
    - map_array: (n_1, ..., n_d, d) in unit coordinates on [0,1]^d
    Returns:
    - new_density: (n_1, ..., n_d) array of nonnegative values
    """
    shape = density.shape
    d = density.ndim
    dtype = density.dtype

    corners = _binary_corners(d)
    num_corners = corners.shape[0]

    map_arr = _coerce_map_array(map_array, shape)

    vertex_shape = tuple(n + 1 for n in shape)
    coords = jnp.indices(vertex_shape, dtype=dtype)
    reshape_pattern = (d,) + (1,) * d
    n_vec_vertex = jnp.array(shape, dtype=dtype).reshape(reshape_pattern)
    unit_coords = coords / n_vec_vertex

    vertex_components = []
    for ax in range(d):
        field = map_arr[..., ax]
        comp = _multilinear_interpolate_cell_center(field, unit_coords, corners)
        vertex_components.append(jnp.clip(comp, 0.0, 1.0))
    vertex_map = jnp.stack(vertex_components, axis=0)  # (d, *vertex_shape)

    def reduce_other_axes(arr, excluded_axis):
        out = arr
        for axis in range(arr.ndim):
            if axis == excluded_axis:
                continue
            out = _adjacent_max(out, axis=axis)
        return out

    stretch_components = [
        reduce_other_axes(jnp.abs(jnp.diff(vertex_map[ax], axis=ax)), ax)
        for ax in range(d)
    ]

    slice_sizes = (2,) * d
    total_cells = density.size
    n_vec_float = jnp.array(shape, dtype=dtype)
    n_vec_int = jnp.array(shape, dtype=jnp.int32)

    def unravel(index):
        idx = index
        coords_rev = []
        for size in reversed(shape):
            size_val = jnp.int32(size)
            coords_rev.append(lax.rem(idx, size_val))
            idx = lax.div(idx, size_val)
        return tuple(reversed(coords_rev))

    def extract_cell_vertices(idx_tuple):
        starts = list(idx_tuple)
        vertices = []
        for ax in range(d):
            block = lax.dynamic_slice(vertex_map[ax], starts, slice_sizes)
            vertices.append(block.reshape(-1))
        return jnp.stack(vertices, axis=1)

    def evaluate_cell_map(cell_vertices, bary):
        bary = jnp.clip(bary, 0.0, 1.0)
        weights = jnp.prod(
            jnp.where(corners == 1, bary, 1.0 - bary),
            axis=1,
        )
        return jnp.sum(weights[:, None] * cell_vertices, axis=0)

    def scatter_sample(rho_array, point, sample_mass):
        scaled = point * n_vec_float - 0.5
        base = jnp.floor(scaled).astype(jnp.int32)
        base = jnp.clip(base, 0, n_vec_int - 1)
        frac = scaled - base.astype(dtype)
        other = base + jnp.sign(frac).astype(jnp.int32)
        other = jnp.clip(other, 0, n_vec_int - 1)
        frac_abs = jnp.abs(frac)
        base_w = 1.0 - frac_abs
        other_w = frac_abs

        def corner_body(m, acc):
            corner = corners[m]
            idx = jnp.where(corner == 1, other, base)
            weight_components = jnp.where(corner == 1, other_w, base_w)
            weight = jnp.prod(weight_components)
            return acc.at[tuple(idx)].add(sample_mass * weight)

        return lax.fori_loop(0, num_corners, corner_body, rho_array)

    def sample_indices(sample_idx, counts_tuple):
        idx = sample_idx
        coords_list = []
        for count in counts_tuple:
            coords_list.append(lax.rem(idx, count))
            idx = lax.div(idx, count)
        return jnp.stack(coords_list, axis=0)

    def process_cell(idx_tuple, mass, rho_array):
        cell_vertices = extract_cell_vertices(idx_tuple)
        counts_list = []
        for ax in range(d):
            stretch_val = stretch_components[ax][idx_tuple]
            est = jnp.ceil(stretch_val * shape[ax]).astype(jnp.int32)
            counts_list.append(jnp.maximum(1, est))
        counts_tuple = tuple(counts_list)
        counts_array = jnp.stack(counts_tuple)
        sample_total = jnp.prod(counts_array)
        sample_total = jnp.maximum(sample_total, 1)
        mass_per_sample = mass / sample_total.astype(dtype)
        counts_float = counts_array.astype(dtype)

        def cond_fn(state):
            return state[0] < sample_total

        def body_fn(state):
            s_idx, rho_curr = state
            local_indices = sample_indices(s_idx, counts_tuple)
            bary = (local_indices.astype(dtype) + 0.5) / counts_float
            point = evaluate_cell_map(cell_vertices, bary)
            rho_next = scatter_sample(rho_curr, point, mass_per_sample)
            return (s_idx + 1, rho_next)

        _, rho_out = lax.while_loop(cond_fn, body_fn, (jnp.int32(0), rho_array))
        return rho_out

    def cell_body(cell_flat_idx, rho_array):
        idx_tuple = unravel(jnp.int32(cell_flat_idx))
        mass = density[idx_tuple]

        def skip_fn(_):
            return rho_array

        def process_fn(_):
            return process_cell(idx_tuple, mass, rho_array)

        return lax.cond(mass <= 0, skip_fn, process_fn, operand=None)

    rho0 = jnp.zeros_like(density)
    rho = lax.fori_loop(0, total_cells, cell_body, rho0)

    target_mean = jnp.mean(density)
    rho_mean = jnp.mean(rho)
    rho = lax.cond(
        rho_mean > 0,
        lambda _: rho * (target_mean / rho_mean),
        lambda _: rho,
        operand=None,
    )

    return rho, None
