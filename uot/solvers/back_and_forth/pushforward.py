from jax import lax
from jax import numpy as jnp

# ------------------------ small n-D utilities ------------------------

def _central_gradient_nd(psi):
    """
    Central differences with edge (Neumann) BC in each axis.
    Returns grad with shape (d, *psi.shape).
    Assumes grid spacing h_i = 1 / n_i on [0,1]^d.
    """
    shape = psi.shape
    d = psi.ndim
    # pad ALL axes by 1 so slicing is consistent
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
        grads.append((forward - backward) / (2.0 * h))

    return jnp.stack(grads, axis=0)   # (d, *shape)

def _cic_pushforward_nd(density, psi):
    shape = density.shape
    d = density.ndim
    n_vec = jnp.array(shape, dtype=jnp.float32)

    grad = _central_gradient_nd(psi)                       # (d, *shape)
    idx  = jnp.indices(shape, dtype=jnp.float32)           # (d, *shape)
    s    = idx + grad * n_vec.reshape((-1,) + (1,) * d)    # (d, *shape)

    base = jnp.floor(s).astype(jnp.int32)
    for ax in range(d):
        base_ax = jnp.clip(base[ax], 0, shape[ax] - 2)     # ensure base+1 valid
        base    = base.at[ax].set(base_ax)
    frac = s - base                                        # (d, *shape)

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
    return out.reshape(shape)
