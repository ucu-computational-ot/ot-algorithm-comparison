import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional

from uot.data.measure import DiscreteMeasure
from .sinkhorn_log import SinkhornTwoMarginalLogJaxSolver

# ---------- helpers ----------

def _normalize_weights(w: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    w = jnp.asarray(w, dtype=jnp.float64)
    w = jnp.clip(w, eps, jnp.inf)
    return w / w.sum()

def _to_discrete_normalized(measure) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Works for DiscreteMeasure (points, weights) or GridMeasure via .to_discrete()."""
    if hasattr(measure, "to_discrete"):
        X, w = measure.to_discrete(include_zeros=False)
    else:
        # assume DiscreteMeasure-like (points, weights) attributes
        X, w = measure.points, measure.weights
    X = jnp.asarray(X, dtype=jnp.float32)
    w = _normalize_weights(jnp.asarray(w))
    return X, w

def _cost_matrix_sqeuclidean(X: jnp.ndarray, Y: jnp.ndarray,
                             batch_size: Optional[int] = None,
                             dtype=jnp.float32) -> jnp.ndarray:
    """
    Build full C = ||x - y||^2. You can set batch_size to limit peak memory.
    Your solver expects a dense array (it renormalizes internally).
    """
    X = jnp.asarray(X, dtype=dtype)
    Y = jnp.asarray(Y, dtype=dtype)
    n, m = X.shape[0], Y.shape[0]
    if batch_size is None:
        # (x^2 + y^2 - 2 x·y) trick
        x2 = jnp.sum(X * X, axis=1, keepdims=True)     # (n,1)
        y2 = jnp.sum(Y * Y, axis=1, keepdims=True).T   # (1,m)
        C = x2 + y2 - 2.0 * (X @ Y.T)
        return jnp.maximum(C, 0.0)
    else:
        # assemble in blocks to save memory
        rows = []
        for i in range(0, n, batch_size):
            Xb = X[i:i+batch_size]
            x2 = jnp.sum(Xb * Xb, axis=1, keepdims=True)      # (bs,1)
            y2 = jnp.sum(Y * Y, axis=1, keepdims=True).T      # (1,m)
            Cb = x2 + y2 - 2.0 * (Xb @ Y.T)
            rows.append(jnp.maximum(Cb, 0.0))
        return jnp.concatenate(rows, axis=0)

def _ot_eps_cost(solver: "SinkhornTwoMarginalLogJaxSolver",
                 X: jnp.ndarray, wX: jnp.ndarray,
                 Y: jnp.ndarray, wY: jnp.ndarray,
                 reg: float, maxiter: int, tol: float,
                 batch_size: Optional[int]) -> float:
    """
    Compute OT_ε(X,Y) using your solver. Returns the 'cost' field (un-normalized units).
    """
    # Build cost matrix (your solver divides by its max internally)
    C = _cost_matrix_sqeuclidean(X, Y, batch_size=batch_size, dtype=jnp.float32)

    mu = DiscreteMeasure(X, wX)
    nu = DiscreteMeasure(Y, wY)

    out = solver.solve(
        marginals=(mu, nu),
        costs=(C,),
        reg=reg,
        maxiter=maxiter,
        tol=tol,
    )
    # out["cost"] is computed as jnp.sum(P * original costs[0]) — that's what we want
    return float(out["cost"])

# ---------- main API ----------

def sinkhorn_divergence_with_solver(
    source,            # GridMeasure or DiscreteMeasure
    target,            # GridMeasure or DiscreteMeasure
    *,
    reg: float = 1e-3, # ε on the (internally normalized) cost; good starting range: 1e-3..1e-2
    maxiter: int = 1000,
    tol: float = 1e-6,
    batch_size: Optional[int] = None,
) -> dict:
    """
    Computes S_ε = OT_ε(μ,ν) - 1/2 OT_ε(μ,μ) - 1/2 OT_ε(ν,ν) with your JAX Sinkhorn solver.
    Returns all three component costs as well.
    """
    solver = SinkhornTwoMarginalLogJaxSolver()

    X, wX = _to_discrete_normalized(source)
    Y, wY = _to_discrete_normalized(target)

    # Cross term
    ot_xy = _ot_eps_cost(solver, X, wX, Y, wY, reg, maxiter, tol, batch_size)

    # Self terms (use the same ε!)
    ot_xx = _ot_eps_cost(solver, X, wX, X, wX, reg, maxiter, tol, batch_size)
    ot_yy = _ot_eps_cost(solver, Y, wY, Y, wY, reg, maxiter, tol, batch_size)

    S_eps = ot_xy - 0.5 * (ot_xx + ot_yy)

    return {
        "sinkhorn_divergence": S_eps,           # S_ε(μ,ν)
        "OT_eps_xy": ot_xy,                     # OT_ε(μ,ν)
        "OT_eps_xx": ot_xx,                     # OT_ε(μ,μ)
        "OT_eps_yy": ot_yy,                     # OT_ε(ν,ν)
        "reg": reg,
        "maxiter": maxiter,
        "tol": tol,
        # optional W2-like scale (not a true metric, but common to report)
        "sinkhorn_divergence_w2_like": float(np.sqrt(max(S_eps, 0.0))),
    }
