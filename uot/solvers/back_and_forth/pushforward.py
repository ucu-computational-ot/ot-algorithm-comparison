from jax import lax
from jax import numpy as jnp


def _central_gradient_nd(psi):
    """
    Central differences with one-sided differences at boundaries.
    Returns grad with shape (d, *psi.shape).
    Assumes grid spacing h_i = 1 / n_i on [0,1]^d.
    """
    shape = psi.shape
    d = psi.ndim
    # pad ALL axes by 1 so slicing is consistent
    # but we will also handle the boundaries separately
    psi_pad = jnp.pad(psi, [(1, 1)] * d, mode='edge')

    grads = []
    for ax in range(d):
        # base slices to pick the central block for non-differenced axes
        center = [slice(1, 1 + n) for n in shape]

        # forward/backward slices along axis ax
        sl_fwd = list(center); sl_fwd[ax] = slice(2, 2 + shape[ax])   # +1 shift
        sl_bwd = list(center); sl_bwd[ax] = slice(0, shape[ax])       # -1 shift

        forward  = psi_pad[tuple(sl_fwd)]
        backward = psi_pad[tuple(sl_bwd)]
        h = 1.0 / shape[ax]           # grid step on [0,1]
        central_diff = (forward - backward) / (2.0 * h)

        # now handle boundaries with one-sided differences
        grad_ax = central_diff
        # left boundary (i=0) with forward difference
        sl_left_fwd = [slice(0, n) if i != ax else slice(1, 2) for i, n in enumerate(shape)]
        sl_left_self = [slice(0, n) if i != ax else slice(0, 1) for i, n in enumerate(shape)]
        left_fwd = psi[tuple(sl_left_fwd)]
        left_self = psi[tuple(sl_left_self)]
        left_diff = (left_fwd - left_self) / h

        # right boundary (i=n-1) with backward difference
        sl_right_bwd = [slice(0, n) if i != ax else slice(-2, -1) for i, n in enumerate(shape)]
        sl_right_self = [slice(0, n) if i != ax else slice(-1, None) for i, n in enumerate(shape)]
        right_bwd = psi[tuple(sl_right_bwd)]
        right_self = psi[tuple(sl_right_self)]
        right_diff = (right_self - right_bwd) / h

        # combine
        sl_left_assign = [slice(None)] * d; sl_left_assign[ax] = 0
        sl_right_assign = [slice(None)] * d; sl_right_assign[ax] = -1
        grad_ax = grad_ax.at[tuple(sl_left_assign)].set(left_diff.squeeze())
        grad_ax = grad_ax.at[tuple(sl_right_assign)].set(right_diff.squeeze())
        grads.append(grad_ax)

    return jnp.stack(grads, axis=0)   # (d, *shape)


def _forward_pushforward_nd(density, psi):
    """
    Pushforward a density field along the flow defined by psi.
    - density: (n_1, n_2, ..., n_d) array of nonnegative values
    - psi:     (n_1, n_2, ..., n_d) array of potential values
    Returns:
    - new_density: (n_1, n_2, ..., n_d) array of nonnegative values
    - grad: (d, n_1, n_2, ..., n_d) array of gradients of psi
    Assumes uniform grid on [0,1]^d with grid spacing h_i = 1/n_i.
    Uses multilinear interpolation (bilinear in 2D, trilinear in 3D, etc).
    Boundary: clamp to [0, n_k - 1] in each dimension.
    0.5*||x||^2 is implicit in the definition of psi.
    1. Compute grad psi
    2. Compute fractional index s = i + grad psi * n
       (vertex-centered interpretation)
    3. Clip s to [0, n_k - 1] in each dimension
    4. Compute base = floor(s) and frac = s - base
    5. Scatter density to 2^d corners with multilinear weights
    """
    shape = density.shape
    d = density.ndim
    n_vec = jnp.array(shape, dtype=jnp.float32)

    grad = _central_gradient_nd(psi)                       # (d, *shape)
    idx  = jnp.indices(shape, dtype=jnp.float32)           # (d, *shape)

    # Compute s, then clip to [0, n_k - 1] per dimension
    # this is the computation for vertex-centered interpretation
    s_raw = idx + grad * n_vec.reshape((-1,) + (1,) * d)  # (d, *shape)
    # this one is for the cell-centered interpretation
    # (but it does not produce ok results somehow)
    # s_raw = idx + 0.5 + grad * n_vec.reshape((-1,) + (1,) * d)  # (d, *shape)
    s = jnp.zeros_like(s_raw)
    for ax in range(d):
        s_ax = jnp.clip(s_raw[ax], 0.0, shape[ax] - 1.0)
        s = s.at[ax].set(s_ax)

    base = jnp.floor(s).astype(jnp.int32)
    # No need for additional clipping on base since s is clipped;
    # base will be in [0, n_k - 2] naturally if s in [0, n_k - 1)
    # But to be safe, clip base
    for ax in range(d):
        base_ax = jnp.clip(base[ax], 0, shape[ax] - 2)
        base = base.at[ax].set(base_ax)
    frac = s - base.astype(jnp.float32)  # Now frac guaranteed in [0,1)

    # --- flatten using static sizes (no int() on tracers) ---
    density_flat = density.reshape(-1)                     # OK: -1 uses static size
    base_flat    = base.reshape(d, -1)
    frac_flat    = frac.reshape(d, -1)

    # row-major strides from static Python shape
    strides_py = []
    p = 1
    for k in range(d-1, -1, -1):
        strides_py.insert(0, p)
        p *= shape[k]
    strides = jnp.array(strides_py, dtype=jnp.int32).reshape(d, 1)  # (d,1)

    out = jnp.zeros_like(density_flat)

    def corner_body(m, out_acc):
        bits = jnp.array([(m >> k) & 1 for k in range(d)], dtype=jnp.int32).reshape(d, 1)
        corner_idx = base_flat + bits                      # (d,N)
        w = jnp.where(bits == 1, frac_flat, 1.0 - frac_flat)
        w = jnp.prod(w, axis=0)                            # (N,)
        flat_idx = jnp.sum(corner_idx * strides, axis=0)   # (N,)
        return out_acc.at[flat_idx].add(density_flat * w)

    out = lax.fori_loop(0, 1 << d, corner_body, out)
    return out.reshape(shape), grad
