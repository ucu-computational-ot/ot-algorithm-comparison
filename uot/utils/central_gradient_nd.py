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