from jax import lax
from jax import numpy as jnp

from .forward_pushforward import _central_gradient_nd
from .pushforward import (
    _binary_corners,
    _multilinear_interpolate_cell_center,
    _adjacent_max,
)
import math

# Reuse your _central_gradient_nd as given in the prompt.

def monge_field_from_psi_nd(psi: jnp.ndarray) -> jnp.ndarray:
    """
    Build the Monge map F(i) = i + grad(psi)(i) * n  on a uniform grid,
    returned in INDEX coordinates.
    Output shape: (d, *psi.shape)
    """
    shape = psi.shape
    d = psi.ndim
    n_vec = jnp.array(shape, dtype=jnp.float32).reshape((d,) + (1,) * d)  # (d,1,...)

    grad = _central_gradient_nd(psi)  # (d, *shape), gradient in physical coords
    idx  = jnp.indices(shape, dtype=jnp.float32)  # (d, *shape)

    # Convert displacement to index units and add to indices
    F = idx + grad * n_vec  # (d, *shape)
    return F


def interp_vector_field_at_positions_nd(V: jnp.ndarray,
                                        s: jnp.ndarray,
                                        keep_inside: bool = True) -> jnp.ndarray:
    """
    Multilinear 'gather' interpolation of a vector field V at positions s
    (both in INDEX coordinates).
      - V: (d, n1, n2, ..., nd)
      - s: (d, n1, n2, ..., nd)   fractional index positions at which to sample V
    Returns:
      - out: (d, n1, n2, ..., nd)  V evaluated at s via multilinear interpolation
    Boundary behavior matches cic_pushforward_nd: clip s to [0, n_k-1],
    clip base to [0, n_k-2], then 2^d-corner weights.
    """
    d = V.shape[0]
    shape = V.shape[1:]
    assert s.shape == (d, *shape), "s must be (d, *shape) in index coords"

    # Per-axis clip of positions into [0, n_k - 1]
    s_clipped = []
    for ax, nk in enumerate(shape):
        s_clipped.append(jnp.clip(s[ax], 0.0, nk - 1.0))
    s = jnp.stack(s_clipped, axis=0)  # (d, *shape)

    base = jnp.floor(s).astype(jnp.int32)  # (d, *shape)
    # Ensure room for the +1 corner on every axis
    for ax, nk in enumerate(shape):
        base_ax = jnp.clip(base[ax], 0, nk - 2)
        base = base.at[ax].set(base_ax)

    frac = s - base.astype(jnp.float32)  # (d, *shape) in [0,1)

    # Flatten everything to (d, N) where N = prod(shape)
    N = 1
    for nk in shape:
        N *= nk

    V_flat   = V.reshape(d, N)          # (d, N)
    base_flat = base.reshape(d, N)      # (d, N)
    frac_flat = frac.reshape(d, N)      # (d, N)

    # Row-major strides for flattening
    strides_py = []
    p = 1
    for k in range(len(shape)-1, -1, -1):
        strides_py.insert(0, p)
        p *= shape[k]
    strides = jnp.array(strides_py, dtype=jnp.int32).reshape(d, 1)  # (d,1)

    out_flat = jnp.zeros_like(V_flat)

    def corner_body(m, acc):
        # Bits ∈ {0,1}^d select the corner
        bits = jnp.array([(m >> k) & 1 for k in range(d)],
                         dtype=jnp.int32).reshape(d, 1)            # (d,1)
        corner_idx = base_flat + bits                               # (d,N)

        # weights: product_k [ frac_k if bit=1 else (1-frac_k) ]
        w_comp = jnp.where(bits == 1, frac_flat, 1.0 - frac_flat)    # (d,N)
        w = jnp.prod(w_comp, axis=0)                                 # (N,)

        # Linearized indices
        flat_idx = jnp.sum(corner_idx * strides, axis=0)             # (N,)

        # Gather: shape (d, N)
        vals = jnp.take(V_flat, flat_idx, axis=1)
        return acc + vals * w  # broadcast (N,) to (d,N)

    out_flat = lax.fori_loop(0, 1 << d, corner_body, out_flat)
    out = out_flat.reshape((d, *shape))

    if keep_inside:
        for ax, nk in enumerate(shape):
            out_ax = jnp.clip(out[ax], 0.0, nk - 1.0)
            out = out.at[ax].set(out_ax)
    return out


def compose_monge_map_nd(T: jnp.ndarray,
                         psi: jnp.ndarray | None = None,
                         F: jnp.ndarray | None = None,
                         keep_inside: bool = True) -> jnp.ndarray:
    """
    Compose maps in INDEX coordinates with consistent clipping/interp:
      - If psi is given: build F = monge_field_from_psi_nd(psi),
        then return F ∘ T evaluated via multilinear gather.
      - Or pass a precomputed F (shape (d, *shape)).
    Shapes:
      T : (d, n1, n2, ..., nd)   -- a map in index coords (per-voxel targets)
      psi: (n1, n2, ..., nd)     -- potential (optional)
      F : (d, n1, n2, ..., nd)   -- vector field map in index coords (optional)

    Returns:
      T_new = (F ∘ T) with CIC-gather interpolation and boundary clamping.
    """
    if (psi is None) == (F is None):
        raise ValueError("Pass exactly one of {psi, F}.")

    if F is None:
        F = monge_field_from_psi_nd(psi)

    # Sanity: T and F must share the same spatial shape
    if T.shape != F.shape:
        raise ValueError(f"Shape mismatch: T{T.shape} vs F{F.shape}")

    # Evaluate F at fractional index positions given by T
    T_new = interp_vector_field_at_positions_nd(F, T, keep_inside=keep_inside)
    return T_new


# Convenience: build a fresh Monge map directly from psi (index coords)
def monge_map_from_psi_nd(psi: jnp.ndarray, keep_inside: bool = True) -> jnp.ndarray:
    """
    Return F(i)=i+grad(psi)(i)*n in INDEX coords, optionally clamped into domain.
    """
    F = monge_field_from_psi_nd(psi)
    if keep_inside:
        for ax, nk in enumerate(psi.shape):
            F = F.at[ax].set(jnp.clip(F[ax], 0.0, nk - 1.0))
    return F


# (Optional) helpers for unit/index coordinate conversion:
def index_to_unit(T: jnp.ndarray) -> jnp.ndarray:
    d = T.shape[0]
    shape = T.shape[1:]
    n_vec = jnp.array(shape, dtype=jnp.float32).reshape((d,) + (1,) * len(shape))
    return T / n_vec

def unit_to_index(U: jnp.ndarray) -> jnp.ndarray:
    d = U.shape[0]
    shape = U.shape[1:]
    n_vec = jnp.array(shape, dtype=jnp.float32).reshape((d,) + (1,) * len(shape))
    return U * n_vec


def map_index_to_unit(T: jnp.ndarray, *, center_offset: bool = False) -> jnp.ndarray:
    """
    Convert a Monge map from index coordinates to unit coordinates on [0,1]^d.
    If center_offset is True, treat index coordinates as cell centers.
    """
    d = T.shape[0]
    shape = T.shape[1:]
    n_vec = jnp.array(shape, dtype=jnp.float32).reshape((d,) + (1,) * len(shape))
    if center_offset:
        return (T + 0.5) / n_vec
    return T / n_vec


def monge_map_cic_from_psi_nd(psi: jnp.ndarray) -> jnp.ndarray:
    """
    Build a Monge map in index coordinates using the same displacement rule
    as cic_pushforward_nd.
    """
    shape = psi.shape
    d = psi.ndim
    n_vec = jnp.array(shape, dtype=jnp.float32).reshape((d,) + (1,) * d)

    grad = _central_gradient_nd(psi)
    idx = jnp.indices(shape, dtype=jnp.float32)
    s_raw = idx + 0.5 + grad * n_vec
    clipped = []
    for ax in range(d):
        clipped.append(jnp.clip(s_raw[ax], 0.0, shape[ax] - 1.0))
    return jnp.stack(clipped, axis=0)


def monge_map_adaptive_from_psi_nd(psi: jnp.ndarray) -> jnp.ndarray:
    """
    Build a Monge map (in index coordinates) that mirrors adaptive_pushforward_nd.
    For each cell we re-evaluate the adaptive sampling procedure and average
    the sample locations to obtain a representative map value.
    """
    shape = psi.shape
    d = psi.ndim
    dtype = psi.dtype
    corners = _binary_corners(d)

    vertex_shape = tuple(n + 1 for n in shape)
    coords = jnp.indices(vertex_shape, dtype=dtype)
    reshape_pattern = (d,) + (1,) * d
    n_vec_vertex = jnp.array(shape, dtype=dtype).reshape(reshape_pattern)
    unit_coords = coords / n_vec_vertex

    vertex_components = []
    for ax in range(d):
        step = 1.0 / shape[ax]
        shifted_plus = unit_coords.at[ax].set(unit_coords[ax] + step)
        shifted_minus = unit_coords.at[ax].set(unit_coords[ax] - step)
        f_plus = _multilinear_interpolate_cell_center(psi, shifted_plus, corners)
        f_minus = _multilinear_interpolate_cell_center(psi, shifted_minus, corners)
        grad = (f_plus - f_minus) / (2.0 * step)
        coord_axis = unit_coords[ax]
        target_coord = jnp.clip(coord_axis + grad, 0.0, 1.0)
        vertex_components.append(target_coord)
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
    total_cells = math.prod(shape)
    n_vec_float = jnp.array(shape, dtype=dtype)

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
        return jnp.stack(vertices, axis=1)  # (num_corners, d)

    def evaluate_cell_map(cell_vertices, bary):
        bary = jnp.clip(bary, 0.0, 1.0)
        weights = jnp.prod(
            jnp.where(corners == 1, bary, 1.0 - bary),
            axis=1,
        )
        return jnp.sum(weights[:, None] * cell_vertices, axis=0)

    def sample_indices(sample_idx, counts_tuple):
        idx = sample_idx
        coords_list = []
        for count in counts_tuple:
            coords_list.append(lax.rem(idx, count))
            idx = lax.div(idx, count)
        return jnp.stack(coords_list, axis=0)

    def cell_body(cell_flat_idx, T):
        idx_tuple = unravel(jnp.int32(cell_flat_idx))
        cell_vertices = extract_cell_vertices(idx_tuple)

        counts_list = []
        for ax in range(d):
            stretch_val = stretch_components[ax][idx_tuple]
            est = jnp.ceil(stretch_val * shape[ax]).astype(jnp.int32)
            counts_list.append(jnp.maximum(1, est))
        counts_tuple = tuple(counts_list)
        counts_array = jnp.stack(counts_tuple)
        sample_total = jnp.maximum(jnp.prod(counts_array), 1)
        counts_float = counts_array.astype(dtype)

        def cond_fn(state):
            return state[0] < sample_total

        def body_fn(state):
            s_idx, acc = state
            local_indices = sample_indices(s_idx, counts_tuple)
            bary = (local_indices.astype(dtype) + 0.5) / counts_float
            point = evaluate_cell_map(cell_vertices, bary)
            return (s_idx + 1, acc + point)

        _, sum_points = lax.while_loop(
            cond_fn,
            body_fn,
            (jnp.int32(0), jnp.zeros((d,), dtype=dtype)),
        )
        mean_point = sum_points / sample_total.astype(dtype)
        map_value = mean_point * n_vec_float - 0.5
        clipped = []
        for ax in range(d):
            clipped.append(jnp.clip(map_value[ax], 0.0, shape[ax] - 1.0))
        map_vec = jnp.stack(clipped, axis=0)
        return T.at[(slice(None),) + idx_tuple].set(map_vec)

    T0 = jnp.zeros((d,) + shape, dtype=dtype)
    return lax.fori_loop(0, total_cells, cell_body, T0)
