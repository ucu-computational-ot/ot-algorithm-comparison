import jax
jax.config.update("jax_enable_x64", True)

from jax import lax
from jax import numpy as jnp
from jax.scipy.fft import dctn, idctn
from functools import partial
from uot.solvers.back_and_forth.c_transform import c_transform_quadratic_fast
from uot.solvers.back_and_forth.forward_pushforward import cic_pushforward_nd

import numpy as np
from matplotlib import pyplot as plt

# %% [markdown]
# # BACK-AND-FORTH

# %%
# ----------------------- small helpers -----------------------

def _dctn(a):   # DCT-II (ortho)
    return dctn(a, type=2, norm="ortho")

def _idctn(a):  # inverse of DCT-II with 'ortho' in SciPy/JAX API
    return idctn(a, type=2, norm="ortho")

# ----------------------- DCT-II n-D helpers -----------------------
# def _dctn(x, axes=None):
#     """Orthonormal DCT-II along given axes (separable)."""
#     if axes is None: axes = tuple(range(x.ndim))
#     y = x
#     for ax in axes:
#         y = dct(y, type=2, axis=ax, norm='ortho')
#     return y

# def _idctn(x, axes=None):
#     """Inverse of _dctn for type=2 with 'ortho'."""
#     if axes is None: axes = tuple(range(x.ndim))
#     y = x
#     for ax in axes:
#         y = idct(y, type=2, axis=ax, norm='ortho')
#     return y

def _r2_from_coords(coords):
    grids = jnp.meshgrid(*coords, indexing="ij")
    r2 = jnp.zeros_like(grids[0])
    for G in grids:
        r2 = r2 + G * G
    return r2

def _initialize_kernel_nd_from_coords(coords, *, use_endpoints=False, dtype=jnp.float32):
    """
    Neumann (-Δ) eigenvalues on a tensor grid defined by `coords`.
    Matches the DCT-II solver below.
    """
    nd = len(coords)
    shape = tuple(len(x) for x in coords)
    lams = []
    for ax, x in enumerate(coords):
        n = x.shape[0]
        h = (x[-1] - x[0]) / (n - 1) if use_endpoints else (x[1] - x[0])
        m = jnp.arange(n, dtype=dtype)

        # DCT-II–friendly spectrum (grid w/o fixed endpoints):
        # lam1d = 2.0 * (1.0 - jnp.cos(jnp.pi * m / n)) / (h * h)
        # If you *do* use endpoints, use this instead:
        lam1d = 4.0 * jnp.sin(jnp.pi * m / (2 * (n - 1)))**2 / (h * h)

        sh = (1,) * ax + (n,) + (1,) * (nd - ax - 1)
        lam1d = lam1d.reshape(sh)
        lams.append(jnp.broadcast_to(lam1d, shape))

    kernel = jnp.add.reduce(jnp.stack(lams, axis=0)).astype(dtype)
    # DC mode: +∞ so division zeroes it
    kernel = kernel.at[(0,) * nd].set(jnp.inf)
    return kernel

def neumann_kernel_nd(shape, lengths, dtype=jnp.float64):
    """
    Eigenvalue tensor Λ for (-Δ_h) with Neumann BC on a cell-centered grid,
    diagonalized by DCT-II. Λ[0,...,0] is set to +∞ to kill the DC mode.
    """
    d = len(shape)
    hs = [L / N for L, N in zip(lengths, shape)]
    parts = []
    
    for i, (N, h) in enumerate(zip(shape, hs)):
        k = jnp.arange(N, dtype=dtype)
        lam1d = (4.0 / (h * h)) * jnp.sin(jnp.pi * k / (2 * N)) ** 2   # = 2(1-cos(pi*k/N))/h^2
        sh = (1,) * i + (N,) + (1,) * (d - i - 1)
        parts.append(jnp.reshape(lam1d, sh))

    # Option A (simple): stack then sum along a new axis
    Lam = jnp.sum(jnp.stack([jnp.broadcast_to(p, shape) for p in parts], axis=0), axis=0).astype(dtype)

    # Option B (lower peak memory): accumulate without stacking
    # Lam = jnp.zeros(shape, dtype=dtype)
    # for p in parts:
    #     Lam = Lam + jnp.broadcast_to(p, shape)

    Lam = Lam.at[(0,) * d].set(jnp.inf)  # remove DC null mode
    return Lam

# %%
# ----------------------- Back-and-Forth (keeps history) -----------------------

@partial(jax.jit, static_argnames=("maxiterations", "progressbar", "stepsize_lower_bound", "error_metric"))
def backnforth_sqeuclidean_nd(
    mu: jnp.ndarray,                 # shape (n0,...,nd-1)
    nu: jnp.ndarray,                 # shape (n0,...,nd-1)
    coordinates: list[jnp.ndarray],  # len d, each length n_k (uniform per axis)
    stepsize: float,
    maxiterations: int,
    tolerance: float,
    progressbar: bool = False,
    pushforward_fn=cic_pushforward_nd, # allow swapping deposition schemes
    stepsize_lower_bound: float = 0.01,
    error_metric: str = 'tv_phi'
):
    """
    Back-and-Forth (quadratic cost) with full debug histories stored each iteration.
    
    error_metric: 'tv_phi' | 'tv_psi' | 'l_inf_psi' | 'h1_psi' | 'sqrt_max_grad' | 'transportation_cost' | 'transportation_cost_relative'

    Conventions:
      T_φ(x) = x - ∇φ(x),  ρ_φ := (T_φ)_# μ  (lives on ν-grid)
      S_ψ(y) = y - ∇ψ(y),  ρ_ψ := (S_ψ)_# ν  (lives on μ-grid)
      φ ← φ + σ (-Δ)^{-1}(ν - ρ_φ),  ψ ← ψ + σ (-Δ)^{-1}(μ - ρ_ψ)
    """

    # --- shapes / dtype
    shape = mu.shape
    d = len(coordinates)
    assert nu.shape == shape and d == mu.ndim == nu.ndim
    for k in range(d):
        assert coordinates[k].shape[0] == shape[k]
    Ls = [coord[-1] for coord in coordinates]
    init_stepsize = stepsize

    coords = tuple(jnp.asarray(c, dtype=mu.dtype) for c in coordinates)

    # --- precompute geometry
    r2 = _r2_from_coords(coords).astype(mu.dtype)
    #     for poisson_neumann solver
    kernel = _initialize_kernel_nd_from_coords(coords, dtype=mu.dtype)
    #     for solve_neumann_poisson_nd_dct2
    Lam = neumann_kernel_nd(shape, Ls, dtype=mu.dtype)
    cell_vol = jnp.prod(jnp.array([c[1] - c[0] for c in coords], dtype=mu.dtype))

    # quadratic c-transform (your fast routine)
    c_transform = partial(c_transform_quadratic_fast, coords_list=coords)

    # Poisson solver: (-Δ) u = rhs (Neumann, zero-mean enforced)
    def solve_neumann_poisson_nd_dct2(f):
        """
        Solve -Δ u = f on a cell-centered grid (Neumann BC), n-D, using DCT-II.
        f: n-D array with shape = grid shape (Ni); must be zero-mean (we enforce it).
        lengths: tuple of Li per axis.
        Returns u with mean(u)=0 gauge.
        """
        f = f - f.mean()  # compatibility
        Fh  = _dctn(f)                      # forward transforms
        Uh  = Fh / Lam                      # divide by eigenvalues
        Uh  = Uh.at[(0,)*f.ndim].set(0.0)   # DC null mode
        u   = _idctn(Uh)                    # back to physical
        return u - u.mean()

    # φ ← φ + σ (-Δ)^{-1}(target − rho); return H^{-1}-energy
    def update_potential(phi, rho, target, sigma):
        r = target - rho
        # u = poisson_neumann(r)
        u = solve_neumann_poisson_nd_dct2(r)
        phi_new = phi + sigma * u
        grad_sq = (jnp.vdot(r, u).real) * cell_vol          # <r, u> = ||∇_{H^1}J||^2
        return phi_new, grad_sq, u, r

    # Dual functional (quadratic)
    def dual_value(phi, psi):
        return cell_vol * (
                0.5 * (r2 * (mu + nu)).sum() - (phi * mu).sum() - (psi * nu).sum()
            )

    # Armijo–Goldstein on the half-step we just performed
    def stepsize_update(sigma, new_val, old_val, grad_sq,
                        # upper=0.90, lower=0.1, scale_down=0.985):
                        # upper=0.75, lower=0.25, scale_down=0.8):
                        upper=0.75, lower=0.25, scale_down=0.95):
        # SIGMA_LOWER_BOUND = stepsize_     # keep stepsizes away from 0
        scale_up = 1.0 / scale_down
        gain = new_val - old_val
        old_sigma = sigma
        sigma = jnp.where(
            gain > sigma * upper * grad_sq,
            sigma * scale_up, sigma)
        sigma = jnp.where(
            gain < sigma * lower * grad_sq,
            sigma * scale_down, sigma)
        # jax.debug.print("[stepsize_update] gain = {}; up = {}; low = {}; sigma {} -> {}",
        #                 gain, grad_sq * sigma * upper, grad_sq * sigma * lower, old_sigma, sigma)
        # do not allow to drop below the set threshold
        # sigma = jnp.maximum(sigma, stepsize_lower_bound)
        sigma = jnp.clip(sigma, stepsize_lower_bound, 2 * init_stepsize)
        return sigma
    
    def _central_grad_axis_neumann(u, axis, h):
        g = jnp.zeros_like(u)
        slc_mid = [slice(None)] * u.ndim
        slc_lo = [slice(None)] * u.ndim
        slc_hi = [slice(None)] * u.ndim
        slc_mid[axis] = slice(1, -1)
        slc_lo[axis] = slice(0, -2)
        slc_hi[axis] = slice(2, None)
        g = g.at[tuple(slc_mid)].set(
            (u[tuple(slc_hi)] - u[tuple(slc_lo)]) / (2.0 * h)
        )
        return g

    def seminorm_dotH1(u, coords):
        h = [c[1] - c[0] for c in coords]
        cell_vol = jnp.prod(jnp.array(h, dtype=u.dtype))
        grad_sq = 0.0
        for ax, c in enumerate(coords):
            g = _central_grad_axis_neumann(u, ax, c[1] - c[0])
            grad_sq = grad_sq + jnp.vdot(g, g).real * cell_vol
        return grad_sq

    # --- init
    phi0 = jnp.zeros_like(mu)
    psi0 = jnp.zeros_like(nu)
    sigma0 = jnp.asarray(stepsize, dtype=mu.dtype)
    D0 = dual_value(phi0, psi0)

    errors0 = jnp.full((maxiterations,), jnp.inf, dtype=mu.dtype)
    dual_values0 = jnp.full((maxiterations,), -jnp.inf, dtype=mu.dtype)

    # full histories (bulky by design)
    phi_after_update0       = jnp.zeros((maxiterations, *shape), dtype=mu.dtype)
    psi_after_update0       = jnp.zeros((maxiterations, *shape), dtype=mu.dtype)
    phi_after_ctransform0   = jnp.zeros((maxiterations, *shape), dtype=mu.dtype)
    psi_after_ctransform0   = jnp.zeros((maxiterations, *shape), dtype=mu.dtype)
    rho_before_phi_update0  = jnp.zeros((maxiterations, *shape), dtype=mu.dtype)  # ρ_φ
    rho_before_psi_update0  = jnp.zeros((maxiterations, *shape), dtype=mu.dtype)  # ρ_ψ
    pde_sol_phi0 = jnp.zeros((maxiterations, *shape), dtype=mu.dtype)
    pde_sol_psi0 = jnp.zeros((maxiterations, *shape), dtype=mu.dtype)
    residual_phi0 = jnp.zeros((maxiterations, *shape), dtype=mu.dtype)
    residual_psi0 = jnp.zeros((maxiterations, *shape), dtype=mu.dtype)
    sigmas = jnp.zeros((maxiterations,), dtype=mu.dtype)

    def body(state):
        (i, phi, psi, sigma, D_old, err, errors, dual_values,
         phi_after_update, psi_after_update,
         phi_after_ctransform, psi_after_ctransform,
         rho_before_phi_update, rho_before_psi_update,
         pde_sols_phi, pde_sols_psi,
         residuals_phi, residuals_psi, sigmas) = state

        # ---- φ half-step ----
        # rho_phi, _ = pushforward_fn(mu, -phi, coords)  # Tφ#μ on ν-grid
        rho_phi, _ = pushforward_fn(mu, -psi)
        # jax.debug.print("L-inf between rho_phi and mu = {}", jnp.max(jnp.abs(rho_phi - mu)))
        # jax.debug.print("L-inf between rho_phi and nu = {}", jnp.max(jnp.abs(rho_phi - nu)))
        rho_before_phi_update = rho_before_phi_update.at[i].set(rho_phi)

        phi, gphi_sq, pde_sol_phi, residual_phi = update_potential(phi, rho_phi, nu, sigma)
        phi_after_update = phi_after_update.at[i].set(phi)
        pde_sols_phi = pde_sols_phi.at[i].set(pde_sol_phi)
        residuals_phi = residuals_phi.at[i].set(residual_phi)

        psi = c_transform(phi)
        psi_after_ctransform = psi_after_ctransform.at[i].set(psi)
        # maybe don't need seconds ctransform
        phi = c_transform(psi)

        # D_mid = dual_value(phi, psi)
        # sigma = stepsize_update(sigma, D_mid, D_old, gphi_sq)

        # ---- ψ half-step ----
        # rho_psi, _ = pushforward_fn(nu, -psi, coords)  # Sψ#ν on μ-grid
        rho_psi, _ = pushforward_fn(nu, -phi)
        rho_before_psi_update = rho_before_psi_update.at[i].set(rho_psi)

        psi, gpsi_sq, pde_sol_psi, residual_psi = update_potential(psi, rho_psi, mu, sigma)
        psi_after_update = psi_after_update.at[i].set(psi)
        pde_sols_psi = pde_sols_psi.at[i].set(pde_sol_psi)
        residuals_psi = residuals_psi.at[i].set(residual_psi)

        phi = c_transform(psi)
        phi_after_ctransform = phi_after_ctransform.at[i].set(phi)
        # maybe don't need seconds ctransform
        psi = c_transform(phi)

        D_new = dual_value(phi, psi)
        # sigma = stepsize_update(sigma, D_new, D_mid, gpsi_sq)
        sigmas = sigmas.at[i].set(sigma)
        # seminorm_dotH1_pde_psi = seminorm_dotH1(pde_sol_psi, coords)
        sigma = lax.cond(i > 0,
                         lambda _: stepsize_update(
                            #  sigma, D_new, D_old, seminorm_dotH1(pde_sols_psi[i-1], coords)),
                             sigma, D_new, D_old, gpsi_sq),
                         lambda _: sigma,
                         operand=None)
        # sigma = stepsize_update(sigma, D_new, D_old, seminorm_dotH1_pde_psi)

        # Parametrized error computation (static branching ensures only one path is compiled)
        if error_metric == 'tv_psi':
            rho_mu, _ = pushforward_fn(mu, -psi)
            err = 0.5 * jnp.sum(jnp.abs(rho_mu - nu))
        elif error_metric == 'tv_phi':
            rho_nu, _ = pushforward_fn(nu, -phi)
            err = 0.5 * jnp.sum(jnp.abs(rho_nu - mu))
        elif error_metric == 'l_inf_psi':
            rho_mu, _ = pushforward_fn(mu, -psi)
            err = jnp.max(jnp.abs(rho_mu - nu))
        elif error_metric == 'h1_psi':
            err = seminorm_dotH1(pde_sol_psi, coords)
        elif error_metric == 'h1_psi_relative':
            err = lax.cond(i > 0,
                  lambda _: jnp.abs(seminorm_dotH1(pde_sol_psi, coords) - seminorm_dotH1(pde_sols_psi[jnp.maximum(i-1,0)], coords)) \
                    / jnp.maximum(seminorm_dotH1(pde_sol_psi, coords), 1e-10),
                  lambda _: jnp.inf,
                  operand=None)
        elif error_metric == 'sqrt_max_grad':
            err = jnp.sqrt(jnp.maximum(gphi_sq, gpsi_sq))
        elif error_metric == 'transportation_cost':
            err = jnp.abs(D_old - D_new)
        elif error_metric == 'transportation_cost_relative':
            err = jnp.abs(D_old - D_new) / jnp.maximum(jnp.abs(D_new), 1e-10)
        else:
            raise ValueError(f"Unknown error_metric: {error_metric}")
        # err = seminorm_dotH1(pde_sol_psi, coords)
        errors = errors.at[i].set(err)
        dual_values = dual_values.at[i].set(D_new)

        if progressbar:
            jax.debug.print("iter {i}: err={e:.3e}, D={D:.6f}, σ={s:.3e}",
                            i=i, e=err, D=D_new, s=sigma)

        return (i + 1, phi, psi, sigma, D_new, err, errors, dual_values,
                phi_after_update, psi_after_update,
                phi_after_ctransform, psi_after_ctransform,
                rho_before_phi_update, rho_before_psi_update,
                pde_sols_phi, pde_sols_psi,
                residuals_phi, residuals_psi, sigmas)

    def cond(state):
        i, *_ = state
        # current error is stored at index i-1
        curr_err = state[6][jnp.maximum(i - 1, 0)]
        # jax.debug.print("cond: i = {}, curr_err = {}", i, curr_err)
        return jnp.logical_and(i < maxiterations, curr_err > tolerance)

    init = (0, phi0, psi0, sigma0, D0, jnp.inf, errors0, dual_values0,
            phi_after_update0, psi_after_update0,
            phi_after_ctransform0, psi_after_ctransform0,
            rho_before_phi_update0, rho_before_psi_update0,
            pde_sol_phi0, pde_sol_psi0,
            residual_phi0, residual_psi0, sigmas)

    state = lax.while_loop(cond, body, init)
    (iterations, phi, psi, _, _, _, errors, dual_values,
     phi_after_update, psi_after_update,
     phi_after_ctransform, psi_after_ctransform,
     rho_before_phi_update, rho_before_psi_update,
     pde_sol_phi0, pde_sol_psi0,
     residual_phi0, residual_psi0, sigmas) = state

    # final pushforwards for reporting
    # rho_mu, _ = pushforward_fn(mu, -phi, coords)  # lives on ν-grid
    # rho_nu, _ = pushforward_fn(nu, -psi, coords)  # lives on μ-grid
    rho_mu, _ = pushforward_fn(mu, -psi)
    rho_nu, _ = pushforward_fn(nu, -phi)

    return (iterations, phi, psi, rho_nu, rho_mu, errors, dual_values,
            phi_after_update, psi_after_update,
            phi_after_ctransform, psi_after_ctransform,
            rho_before_phi_update, rho_before_psi_update,
            pde_sol_phi0, pde_sol_psi0,
            residual_phi0, residual_psi0, sigmas)
