from collections.abc import Sequence
import jax
from jax import lax
import jax.numpy as jnp

from uot.data.measure import GridMeasure
from uot.utils.types import ArrayLike
from uot.solvers.base_solver import BaseSolver

from .method import backnforth_sqeuclidean_nd
from .pushforward import _central_gradient_nd, _cic_pushforward_nd


class BackNForthSqEuclideanSolver(BaseSolver):
    def __init__(self):
        return super().__init__()

    def solve(
        self,
        marginals: Sequence[GridMeasure],
        costs: Sequence[ArrayLike],
        maxiter: int = 1000,
        tol: float = 1e-6,
        stepsize: float = 1,
    ) -> dict:
        if len(marginals) != 2:
            raise ValueError("Back-and-Forth solver accepts only two marginals.")

        mu, nu = marginals[0], marginals[1]
        axes_mu, mu_nd = mu.for_grid_solver(backend="jax", dtype=jnp.float64)
        axes_nu, nu_nd = nu.for_grid_solver(backend="jax", dtype=jnp.float64)

        iters, phi, psi, rho_nu, rho_mu, errs, duals = backnforth_sqeuclidean_nd(
            mu=mu_nd,
            nu=nu_nd,
            coordinates=axes_mu,
            stepsize=stepsize,
            maxiterations=maxiter,
            tolerance=tol,
        )

        # ----- grid helpers -----
        def _grid_spacings(axes):
            # assumes monotone axes, uniform per axis
            hs = [ax[1] - ax[0] if ax.shape[0] > 1 else 1.0 for ax in axes]
            return hs, jnp.prod(jnp.asarray(hs))

        hs, dV = _grid_spacings(axes_mu)

        grids = jnp.meshgrid(*axes_mu, indexing="ij")     # list of d arrays, each (*shape)
        X = jnp.stack(grids, axis=-1)                     # (*shape, d)

        # ----- current monge_map computation (kept as-is) -----
        grad_psi = _central_gradient_nd(psi)              # (d, *shape)
        if grad_psi.shape[0] == len(axes_mu):
            grad_psi = jnp.moveaxis(grad_psi, 0, -1)      # -> (*shape, d)
        monge_map = grad_psi

        if monge_map.shape != X.shape:
            raise ValueError(f"Monge map shape {monge_map.shape} != grid shape {X.shape}")

        diff = X - monge_map
        cost = jnp.sum(jnp.sum(diff * diff, axis=-1) * mu_nd)

        # marginal L2 (your existing diagnostics)
        marginal_L2_mu_to_nu = jnp.linalg.norm((rho_mu - nu_nd).ravel()) * jnp.sqrt(dV)
        marginal_L2_nu_to_mu = jnp.linalg.norm((rho_nu - mu_nd).ravel()) * jnp.sqrt(dV)

        extra = self._extra_metrics(
            mu_nd=mu_nd,
            nu_nd=nu_nd,
            axes_mu=axes_mu,
            X=X,
            psi=psi,
            grad_psi_moved=grad_psi,   # (*shape, d)
            rho_mu=rho_mu,
            rho_nu=rho_nu,
            dV=dV,
        )

        # ----- assemble result -----
        out = {
            "monge_map": monge_map,
            "cost": cost,
            "u_final": phi,
            "v_final": psi,
            "iterations": iters,
            "error": errs[iters - 1],
            "marginal_error": jnp.linalg.norm((rho_mu - nu_nd).ravel()),
            "marginal_error_mu_to_nu": marginal_L2_mu_to_nu,
            "marginal_error_nu_to_mu": marginal_L2_nu_to_mu,
        }
        out.update(extra)
        return out

    # ------------------------------------------------------------------
    # Extra diagnostics: push-forward TV and Monge–Ampère residual
    # ------------------------------------------------------------------
    def _extra_metrics(
        self,
        *,
        mu_nd: jnp.ndarray,
        nu_nd: jnp.ndarray,
        axes_mu,
        X: jnp.ndarray,
        psi: jnp.ndarray,
        grad_psi_moved: jnp.ndarray,   # shape (*shape, d)
        rho_mu: jnp.ndarray,
        rho_nu: jnp.ndarray,
        dV: float,
    ) -> dict:
        """
        Computes:
          - rho_mu_push (CIC), tv_mu_to_nu
          - Monge–Ampère residual norms and det(J) diagnostics

        NOTE on convention:
          We use T(x) = x - ∇psi(x) INSIDE this method for consistency of TV/MA checks.
          Your `monge_map` variable in `solve` is left as-is (grad_psi).
        """
        # Consistent Monge map for metrics
        T = X - grad_psi_moved                                # (*shape, d)

        # (A) Push-forward TV via your CIC (uses -psi to match T = x - ∇psi)
        rho_mu_push = _cic_pushforward_nd(mu_nd, -psi)        # same grid as nu_nd

        # mass-fix to compare probabilities
        mass_mu_push = jnp.sum(rho_mu_push) * dV
        mass_nu = jnp.sum(nu_nd) * dV
        rho_mu_push = rho_mu_push * (mass_nu / jnp.maximum(1e-30, mass_mu_push))

        tv_mu_to_nu = 0.5 * jnp.sum(jnp.abs(rho_mu_push - nu_nd)) * dV

        # (B) Monge–Ampère residual
        Hpsi = self._hessian_via_fd(psi)                      # (d, d, *shape)
        d = psi.ndim
        I = jnp.eye(d, dtype=psi.dtype).reshape(d, d, *([1] * d))
        J = I - Hpsi                                          # Jacobian of T

        detJ = jnp.linalg.det(J.reshape(d, d, -1).transpose(2, 0, 1)).reshape(psi.shape)

        # sample rho_nu at mapped points T(x)
        rho_nu_at_T = self._linear_sample_nd(nu_nd, T, axes_mu)

        ma_residual = rho_nu_at_T * detJ - mu_nd
        ma_L1 = jnp.sum(jnp.abs(ma_residual)) * dV
        ma_L2 = jnp.linalg.norm(ma_residual.ravel()) * jnp.sqrt(dV)
        mu_mass = jnp.sum(mu_nd) * dV
        ma_L1_rel = ma_L1 / jnp.maximum(1e-30, mu_mass)

        # det(J) sanity stats
        detJ_min = jnp.min(detJ)
        detJ_max = jnp.max(detJ)
        detJ_neg_frac = jnp.mean((detJ < 0).astype(jnp.float32))

        return {
            "rho_mu_push": rho_mu_push,
            "tv_mu_to_nu": tv_mu_to_nu,
            "ma_residual_L1": ma_L1,
            "ma_residual_L1_rel": ma_L1_rel,
            "ma_residual_L2": ma_L2,
            "detJ_min": detJ_min,
            "detJ_max": detJ_max,
            "detJ_neg_frac": detJ_neg_frac,
        }

    # ---------------- tiny helpers (stay inside the class) ----------------
    @staticmethod
    def _hessian_via_fd(psi: jnp.ndarray) -> jnp.ndarray:
        """
        Build Hessian by reusing your central-diff gradient:
          H[i,j,...] = ∂^2 psi / ∂x_i ∂x_j
        Same [0,1]^d, h_i = 1/n_i convention as _central_gradient_nd.
        """
        g = _central_gradient_nd(psi)                         # (d, *shape)
        H_list = [_central_gradient_nd(g[i]) for i in range(g.shape[0])]
        H = jnp.stack(H_list, axis=0)                         # (d, d, *shape)
        return H

    @staticmethod
    def _linear_sample_nd(arr: jnp.ndarray, positions: jnp.ndarray, axes) -> jnp.ndarray:
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
