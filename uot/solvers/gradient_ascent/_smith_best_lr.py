import jax.numpy as jnp

def best_lr(
    lrs: jnp.ndarray,
    losses: jnp.ndarray,
    window: int = 5,
    factor: float = 10.0,
    smooth: bool = True,
) -> float:
    """
    Pick the “best” learning-rate from an LR-finder run.

    Parameters
    ----------
    lrs : jnp.ndarray
        1-D array of learning-rates returned by the finder (shape = [N]).
    losses : jnp.ndarray
        1-D array of the corresponding *negative* dual objective
        (i.e. the loss that the finder records).  Shape = [N].
    window : int, optional
        Half-width of the moving-average smoother (default = 5).
        Set ``window=0`` to skip smoothing.
    factor : float, optional
        Safety factor (3-10 in the paper).  The function returns
        ``lr_opt / factor``.  Larger factor → more conservative.
    smooth : bool, optional
        Whether to smooth the loss curve before computing slopes.

    Returns
    -------
    float
        Recommended (conservative) learning-rate.
    """
    # ------------------------------------------------------------------
    # 1. (optional) smooth the loss curve
    # ------------------------------------------------------------------
    if smooth and window > 0:
        # pad on both sides so the convolution does not shrink the array
        pad = jnp.full(window, losses[0])
        smoothed = jnp.convolve(
            jnp.concatenate([pad, losses, pad]),
            jnp.ones(2 * window + 1) / (2 * window + 1),
            mode="valid",
        )
    else:
        smoothed = losses.copy()

    # ------------------------------------------------------------------
    # 2. compute per-step slopes on the *log10* learning-rate axis
    # ------------------------------------------------------------------
    log_lr = jnp.log10(lrs)               # log-scale is what Smith uses
    # slopes[i] = Δloss / Δlog10(lr)  for step i → i+1
    slopes = (smoothed[:-1] - smoothed[1:]) / (log_lr[1:] - log_lr[:-1])

    # ------------------------------------------------------------------
    # 3. the “best” point is the *largest negative slope*
    # ------------------------------------------------------------------
    k_opt = jnp.argmax(slopes)            # index of steepest descent
    lr_opt = lrs[k_opt]                  # raw learning-rate at that point

    # ------------------------------------------------------------------
    # 4. apply the safety factor
    # ------------------------------------------------------------------
    return lr_opt / factor