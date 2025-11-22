from jax import lax
from jax import numpy as jnp
from uot.utils.central_gradient_nd import _central_gradient_nd


def cic_pushforward_nd(density, psi):
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
    # (but it does not produce ok results somehow) update -> produces ok results now :-)
    s_raw = idx + 0.5 + grad * n_vec.reshape((-1,) + (1,) * d)  # (d, *shape)
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


# Backwards compatibility for existing references
_forward_pushforward_nd = cic_pushforward_nd
