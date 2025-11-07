from collections.abc import Sequence

import jax
from jax import lax
import jax.numpy as jnp

from uot.utils.types import ArrayLike

def linear_sample_nd(arr: jnp.ndarray, positions: jnp.ndarray, axes: Sequence[ArrayLike]) -> jnp.ndarray:
    """
    Multilinear interpolation of 'arr' at 'positions' (physical coords).
    positions: (*shape, d) in the same physical units as 'axes'.
    """
    d = arr.ndim
    shape = arr.shape
    assert positions.shape[-1] == d

    h = jnp.array([(ax[1] - ax[0]) if ax.shape[0] > 1 else 1.0 for ax in axes], dtype=arr.dtype)
    x0 = jnp.array([ax[0] for ax in axes], dtype=arr.dtype)

    # physical -> index coords
    s = (positions - x0) / h                              # (*shape, d)
    s = jnp.moveaxis(s, -1, 0)                            # (d, *shape)

    # clamp so base+1 is valid
    eps = 1e-6
    for i in range(d):
        s_i = jnp.clip(s[i], 0.0, shape[i] - 1.0 - eps)
        s = s.at[i].set(s_i)

    base = jnp.floor(s).astype(jnp.int32)                 # (d, *shape)
    frac = s - base                                       # (d, *shape)

    arr_flat = arr.reshape(-1)
    base_flat = base.reshape(d, -1)
    frac_flat = frac.reshape(d, -1)

    # row-major strides
    strides = []
    p = 1
    for k in range(d - 1, -1, -1):
        strides.insert(0, p)
        p *= shape[k]
    strides = jnp.array(strides, dtype=jnp.int32).reshape(d, 1)  # (d,1)

    def corner_value(m, acc):
        bits = jnp.array([(m >> k) & 1 for k in range(d)], dtype=jnp.int32).reshape(d, 1)
        corner_idx = base_flat + bits
        w = jnp.where(bits == 1, frac_flat, 1.0 - frac_flat)
        w = jnp.prod(w, axis=0)                                # (N,)
        flat_idx = jnp.sum(corner_idx * strides, axis=0)       # (N,)
        return acc + w * arr_flat[flat_idx]

    N = base_flat.shape[1]
    out = jnp.zeros((N,), dtype=arr.dtype)
    out = lax.fori_loop(0, 1 << d, corner_value, out)
    return out.reshape(shape)