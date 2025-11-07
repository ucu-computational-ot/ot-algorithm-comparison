# import jax
# from jax import lax
# from jax import numpy as jnp
# from jax.scipy.fft import dctn, idctn
# from functools import partial
# from .c_transform import c_transform_quadratic_fast
# from .pushforward import _forward_pushforward_nd


# # ------------------------ main BFM (d-dimensional) ------------------------
# @partial(jax.jit, static_argnames=('maxiterations', 'progressbar', 'stepsize_lower_bound', 'error_metric'))
# def backnforth_sqeuclidean_nd(
#         mu: jnp.ndarray,                 # shape (n0,...,nd-1)
#         nu: jnp.ndarray,                 # shape (n0,...,nd-1)
#         coordinates: list[jnp.ndarray],  # len d, each length n_k
#         stepsize: float,
#         maxiterations: int,
#         tolerance: float,
#         progressbar: bool = False,
#         pushforward_fn=_forward_pushforward_nd,
#         stepsize_lower_bound: float = 0.01,
#         error_metric: str = 'h1_psi',
#     ):
#     """
#     Dimension-agnostic BFM with quadratic cost on a uniform tensor grid in [0,1]^d.

#     error_metric: 'tv_psi' | 'tv_phi' | 'l_inf_psi' | 'h1_psi' | 'h1_psi_relative' | 'transportation_cost' | 'transportation_cost_relative'
#     """

#     # checks (lightweight; keep in Python tracer-friendly)
#     shape = mu.shape
#     assert nu.shape == shape
#     d = len(coordinates)
#     assert d == mu.ndim == nu.ndim
#     for k in range(d):
#         assert coordinates[k].shape[0] == shape[k]
#     Ls = [coord[-1] for coord in coordinates]
#     init_stepsize = stepsize

#     # c-transform for quadratic cost (will call your fast implementation)
#     c_transform = partial(c_transform_quadratic_fast, coords_list=coordinates)

#     # precompute kernel and r^2 grid
#     kernel = neumann_kernel_nd(shape, Ls, dtype=mu.dtype)
#     r2 = _r2_from_coords(coordinates)
#     cell_vol = jnp.prod(jnp.array([c[1] - c[0] for c in coordinates], dtype=mu.dtype))

#     def dct_neumann_poisson(f):
#         """
#         Solve -Δu = f with Neumann BC via DCT-II (up to constant)
#         on cell-centered grid.
#         """
#         f = f - f.mean()
#         Fh = _dctn(f)
#         Uh = Fh / kernel
#         Uh = Uh.at[(0,)*f.ndim].set(0.0)    # DC null mode
#         u = _idctn(Uh)
#         return u - u.mean()

#     # φ ← φ + σ Δ^{-1}(ρ−ν), return ⟨(-Δ)^{-1}(ρ−ν), (ρ−ν)⟩
#     def update_potential(phi, rho, target, sigma):
#         rho_diff = target - rho
#         recov = dct_neumann_poisson(rho_diff)
#         new_phi = phi + sigma * recov
#         grad_sq = cell_vol * jnp.vdot(rho_diff, recov).real
#         return new_phi, recov, grad_sq

#     # Dual objective (quadratic cost):  ½∫|x|² (μ+ν) - ∫ν φ - ∫μ ψ
#     def dual_value(phi, psi, mu, nu):
#         return cell_vol * jnp.sum(0.5 * r2 * (mu + nu) - mu * phi - nu * psi)
#         # return cell_vol * jnp.sum(0.5 * r2 * (mu + nu) - nu * phi - mu * psi)

#     # Armijo–Goldstein heuristic
#     def stepsize_update(sigma, value, old_value, grad_sq,
#                         upper=0.90, lower=0.10, scale_down=0.985):
#                         # upper=0.75, lower=0.25, scale_down=0.985):  
#         scale_up = 1.0 / scale_down
#         gain = value - old_value
#         old_sigma = sigma
#         sigma = jnp.where(
#             gain > sigma * upper * grad_sq,
#             sigma * scale_down, sigma
#         )
#         sigma = jnp.where(
#             gain < sigma * lower * grad_sq,
#             sigma * scale_up, sigma
#         )
# #
# # cond: i = 0, curr_err = inf
# # cond: i = 1, curr_err = 0.3949669205540193
# # [stepsize_update] gain = 3.58678372897656e-05; up = 2.6978734109075907e-06; low = 2.9976371232306567e-07; sigma 985.514151034625 -> 970.7314387691056
# # cond: i = 2, curr_err = 0.11394873914265198
# # [stepsize_update] gain = 4.1732303535220124e-06; up = 7.623039590090016e-07; low = 8.470043988988907e-08; sigma 970.7314387691056 -> 956.170467187569
# # cond: i = 3, curr_err = 0.013084485991561241
# #
# # iter 0: err=8.095e-01, D=0.000093, σ=9.855e+02
# # [stepsize_update] gain = -8.56163071772638e-06; up = 2.780667794488484e-06; low = 3.0896308827649825e-07; sigma 985.514151034625 -> 1000.5219807458121
# # iter 1: err=1.011e-01, D=0.000085, σ=1.001e+03
#         jax.debug.print("[stepsize_update] gain = {}; up = {}; low = {}; sigma {} -> {}",
#                         gain, grad_sq * sigma * upper, grad_sq * sigma * lower, old_sigma, sigma)
#         sigma = jnp.clip(sigma, stepsize_lower_bound, 2 * init_stepsize)
#         return sigma

#     def body(state):
#         (i, phi, psi, D_value_old, sigma, last_gradient_seminorms, errors, dual_values) = state

#         # --- φ half-step ---
#         # pushforward (ψ acts on μ)
#         rho_phi, _ = pushforward_fn(mu, -psi)
#         phi, pde_sol_phi, grad_sq_phi = update_potential(phi, rho_phi, nu, sigma)
#         psi = c_transform(phi)
#         # phi = c_transform(psi)
#         D_value = dual_value(phi, psi, mu, nu)
#         # update stepsize after update on J(φ)
#         sigma = lax.cond(i > 0,
#                          lambda _: stepsize_update(
#                             sigma, D_value, D_value_old, last_gradient_seminorms['J']),
#                          lambda _: sigma,
#                          operand=None)
#         D_value_old = D_value
#         last_gradient_seminorms['J'] = grad_sq_phi

#         # --- ψ half-step ---
#         # pushforward (φ acts on ν)
#         rho_psi, _ = pushforward_fn(nu, -phi)
#         psi, pde_sol_psi, grad_sq_psi = update_potential(psi, rho_psi, mu, sigma)
#         phi = c_transform(psi)
#         # psi = c_transform(phi)
#         D_value = dual_value(phi, psi, mu, nu)
#         # update stepsize after update on I(ψ)
#         sigma = lax.cond(i > 0,
#                          lambda _: stepsize_update(
#                             # sigma, D_value, D_value_old, grad_sq_psi),
#                             sigma, D_value, D_value_old, last_gradient_seminorms['I']),
#                          lambda _: sigma,
#                          operand=None)

#         # parametrized error computation (static branching->one path compiled)
#         if error_metric == 'tv_psi':
#             rho_mu, _ = pushforward_fn(mu, -psi)
#             err = 0.5 * jnp.sum(jnp.abs(rho_mu - nu))
#         elif error_metric == 'tv_phi':
#             rho_nu, _ = pushforward_fn(nu, -phi)
#             err = 0.5 * jnp.sum(jnp.abs(rho_nu - mu))
#         elif error_metric == 'l_inf_psi':
#             rho_mu, _ = pushforward_fn(mu, -psi)
#             err = jnp.max(jnp.abs(rho_mu - nu))
#         elif error_metric == 'h1_psi':
#             err = grad_sq_psi
#         elif error_metric == 'h1_psi_relative':
#             err = jnp.abs(
#                 last_gradient_seminorms['I'] - grad_sq_psi) / jnp.maximum(
#                 grad_sq_psi, 1e-10)
#         elif error_metric == 'transportation_cost':
#             err = jnp.abs(D_value_old - D_value)
#         elif error_metric == 'transportation_cost_relative':
#             err = jnp.abs(D_value_old - D_value) / jnp.maximum(
#                 jnp.abs(D_value), 1e-10)
#         else:
#             raise ValueError(f"Unknown error_metric: {error_metric}")
#         errors = errors.at[i].set(err)

#         if progressbar:
#             jax.debug.print("iter {i}: err={e:.3e}, D={D:.6f}, σ={s:.3e}",
#                             i=i, e=err, D=D_value, s=sigma)

#         last_gradient_seminorms['I'] = grad_sq_psi
#         dual_values = dual_values.at[i].set(D_value)
#         return (i+1, phi, psi, D_value, sigma, last_gradient_seminorms, errors, dual_values)

#     def cond(state):
#         i = state[0]
#         curr_error = state[6][jnp.maximum(i - 1, 0)]
#         return (i < maxiterations) & (curr_error > tolerance)

#     # init
#     phi0 = jnp.zeros_like(mu)
#     psi0 = jnp.zeros_like(nu)
#     dual0 = dual_value(phi0, psi0, mu, nu)
#     gradient_seminorms0 = {'J': -jnp.inf, 'I': jnp.inf}
#     errors0 = jnp.full((maxiterations,), jnp.inf, dtype=mu.dtype)
#     dual_values0 = jnp.full((maxiterations,), -jnp.inf, dtype=mu.dtype)
#     init = (
#         0, phi0, psi0, dual0, stepsize,
#         gradient_seminorms0, errors0, dual_values0)

#     state = lax.while_loop(cond, body, init)
#     iterations, phi, psi, _, _, _, errors, dual_values = state

#     rho_mu, _ = pushforward_fn(mu, -psi)
#     rho_nu, _ = pushforward_fn(nu, -phi)
#     # return iterations, -phi, -psi, rho_nu, rho_mu, errors, dual_values
#     return iterations, phi, psi, rho_nu, rho_mu, errors, dual_values


# def _r2_from_coords(coords):
#     """
#     coords: list of d arrays with lengths n_k, assumed uniform per axis.
#     Returns r2 grid with shape (n0,...,nd-1), computed as sum_i x_i^2.
#     """
#     grids = jnp.meshgrid(*coords, indexing="ij")
#     r2 = jnp.zeros_like(grids[0])
#     for G in grids:
#         r2 = r2 + G * G
#     return r2


# def _dctn(a):
#     return dctn(a, type=2, norm="ortho")


# def _idctn(a):
#     return idctn(a, type=2, norm="ortho")


# def neumann_kernel_nd(shape, lengths, dtype=jnp.float64):
#     """
#     Eigenvalue tensor Λ for (-Δ_h) with Neumann BC on a cell-centered grid,
#     diagonalized by DCT-II. Λ[0,...,0] is set to +∞ to kill the DC mode.
#     """
#     d = len(shape)
#     hs = [L / N for L, N in zip(lengths, shape)]
#     parts = []
    
#     for i, (N, h) in enumerate(zip(shape, hs)):
#         k = jnp.arange(N, dtype=dtype)
#         lam1d = (4.0 / (h * h)) * jnp.sin(jnp.pi * k / (2 * N)) ** 2   # = 2(1-cos(pi*k/N))/h^2
#         sh = (1,) * i + (N,) + (1,) * (d - i - 1)
#         parts.append(jnp.reshape(lam1d, sh))

#     # Option A (simple): stack then sum along a new axis
#     Lam = jnp.sum(jnp.stack([jnp.broadcast_to(p, shape) for p in parts], axis=0), axis=0).astype(dtype)

#     # Option B (lower peak memory): accumulate without stacking
#     # Lam = jnp.zeros(shape, dtype=dtype)
#     # for p in parts:
#     #     Lam = Lam + jnp.broadcast_to(p, shape)

#     Lam = Lam.at[(0,) * d].set(jnp.inf)  # remove DC null mode
#     return Lam


# ===========================================================================
import jax
# jax.config.update("jax_enable_x64", True)

from jax import lax
from jax import numpy as jnp
from jax.scipy.fft import dctn, idctn
from functools import partial
from uot.solvers.back_and_forth.c_transform import c_transform_quadratic_fast
from uot.solvers.back_and_forth.pushforward import _forward_pushforward_nd

import numpy as np
from matplotlib import pyplot as plt


# ----------------------- DCT helpers -----------------------
def _dctn(a):
    """DCT-II (ortho)"""
    return dctn(a, type=2, norm="ortho")


def _idctn(a):
    """Inverse of DCT-II with 'ortho' in SciPy/JAX API"""
    return idctn(a, type=2, norm="ortho")


# ----------------------- Geometry helpers -----------------------
def _r2_from_coords(coords):
    """Compute squared distance from origin for each grid point."""
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

        # DCT-II–friendly spectrum for endpoints vs. cell-centered
        if use_endpoints:
            lam1d = 4.0 * jnp.sin(jnp.pi * m / (2 * (n - 1)))**2 / (h * h)
        else:
            lam1d = 2.0 * (1.0 - jnp.cos(jnp.pi * m / n)) / (h * h)

        sh = (1,) * ax + (n,) + (1,) * (nd - ax - 1)
        lam1d = lam1d.reshape(sh)
        lams.append(jnp.broadcast_to(lam1d, shape))

    kernel = jnp.add.reduce(jnp.stack(lams, axis=0)).astype(dtype)
    # DC mode: +∞ so division zeroes it
    kernel = kernel.at[(0,) * nd].set(jnp.inf)
    # kernel = kernel.at[(0,) * nd].set(0.0)
    return kernel


def neumann_kernel_nd(shape, lengths, dtype=jnp.float64):
    """
    Eigenvalue tensor Λ for (-Δ_h) with Neumann BC on a cell-centered grid,
    diagonalized by DCT-II. Λ[0,...,0] is set to +∞ to kill the DC mode.
    """
    d = len(shape)
    hs = [L / N for L, N in zip(lengths, shape)]

    # xx, yy = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
    # kernel = 2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy))
    # kernel[0,0] = 1     # to avoid dividing by zero
    
    # Compute eigenvalues for each dimension
    Lam = jnp.zeros(shape, dtype=dtype)
    for i, (N, h) in enumerate(zip(shape, hs)):
        k = jnp.arange(N, dtype=dtype)
        lam1d = (4.0 / (h * h)) * jnp.sin(jnp.pi * k / (2 * N)) ** 2
        
        # Reshape for broadcasting
        sh = (1,) * i + (N,) + (1,) * (d - i - 1)
        lam1d_reshaped = jnp.reshape(lam1d, sh)
        Lam = Lam + jnp.broadcast_to(lam1d_reshaped, shape)

    # Remove DC null mode
    Lam = Lam.at[(0,) * d].set(jnp.inf)
    # Lam = Lam.at[(0,) * d].set(1.0)
    return Lam


# ----------------------- Back-and-Forth Implementation -----------------------
@partial(jax.jit, static_argnames=("maxiterations", "progressbar", "stepsize_lower_bound", "error_metric"))
def backnforth_sqeuclidean_nd(
    mu: jnp.ndarray,                 # shape (n0,...,nd-1)
    nu: jnp.ndarray,                 # shape (n0,...,nd-1)
    coordinates: list[jnp.ndarray],  # len d, each length n_k (uniform per axis)
    stepsize: float,
    maxiterations: int,
    tolerance: float,
    progressbar: bool = False,
    pushforward_fn=_forward_pushforward_nd,
    stepsize_lower_bound: float = 0.01,
    error_metric: str = 'tv_phi'
):
    """
    Back-and-Forth algorithm for quadratic optimal transport with full debug histories.
    
    Parameters:
    -----------
    mu, nu : jnp.ndarray
        Source and target measures on the same grid
    coordinates : list[jnp.ndarray]
        Grid coordinates for each dimension
    stepsize : float
        Initial stepsize for gradient updates
    maxiterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance
    error_metric : str
        One of: 'tv_phi', 'tv_psi', 'l_inf_psi', 'h1_psi', 'h1_psi_relative',
                'sqrt_max_grad', 'transportation_cost', 'transportation_cost_relative'
    
    Returns:
    --------
    Tuple containing iteration count, final potentials, pushforwards, errors, and histories
    """
    
    # Validate inputs
    shape = mu.shape
    d = len(coordinates)
    assert nu.shape == shape and d == mu.ndim == nu.ndim
    for k in range(d):
        assert coordinates[k].shape[0] == shape[k]
    
    # Setup geometry
    Ls = [coord[-1] for coord in coordinates]
    init_stepsize = stepsize
    coords = tuple(jnp.asarray(c, dtype=mu.dtype) for c in coordinates)
    
    # Precompute geometry for dual functional
    r2 = _r2_from_coords(coords).astype(mu.dtype)
    
    # Precompute Poisson solver components
    Lam = neumann_kernel_nd(shape, Ls, dtype=mu.dtype)
    cell_vol = jnp.prod(jnp.array([c[1] - c[0] for c in coords], dtype=mu.dtype))
    mu_nu_grid_sum = 0.5 * (r2 * (mu + nu)).sum()
    
    # Initialize c-transform
    c_transform = partial(c_transform_quadratic_fast, coords_list=coords)
    
    def solve_neumann_poisson_nd_dct2(f):
        """
        Solve -Δ u = f on a cell-centered grid with Neumann BC using DCT-II.
        Enforces zero-mean condition on both input and output.
        """
        f = f - f.mean()  # Ensure zero mean
        Fh = _dctn(f)     # Forward transform
        Uh = Fh / Lam     # Divide by eigenvalues
        Uh = Uh.at[(0,)*f.ndim].set(0.0)  # Kill DC mode
        u = _idctn(Uh)    # Back to physical space
        # return u
        return u - u.mean()  # Ensure zero mean result
    
    def update_potential(phi, rho, target, sigma):
        """Update potential: φ ← φ + σ (-Δ)^{-1}(target - rho)"""
        residual = target - rho
        pde_solution = solve_neumann_poisson_nd_dct2(residual)
        phi_new = phi + sigma * pde_solution
        grad_squared = (jnp.vdot(residual, pde_solution).real) * cell_vol
        return phi_new, grad_squared, pde_solution, residual
    
    def dual_value(phi, psi):
        """Compute dual functional value for quadratic cost."""
        return cell_vol * (
            mu_nu_grid_sum - (phi * mu).sum() - (psi * nu).sum()
        )
    
    def stepsize_update(sigma, new_val, old_val, grad_sq,
                    #    upper=0.80, lower=0.25, scale_down=0.85):
                       upper=0.75, lower=0.25, scale_down=0.95):
                    #    upper=0.9, lower=0.1, scale_down=0.985):
        """Armijo-Goldstein stepsize update."""
        scale_up = 1.0 / scale_down
        gain = new_val - old_val
        old_sigma = sigma
        
        # Update stepsize based on gain
        sigma = jnp.where(gain > sigma * upper * grad_sq, sigma * scale_up, sigma)
        sigma = jnp.where(gain < sigma * lower * grad_sq, sigma * scale_down, sigma)
        
        if progressbar:
            pass
            # jax.debug.print(
            #     "[stepsize_update] gain = {}; up = {}; low = {}; sigma {} -> {}",
            #     gain, grad_sq * sigma * upper, grad_sq * sigma * lower, old_sigma, sigma
            # )
        
        # Clip to reasonable bounds
        # sigma = jnp.clip(sigma, stepsize_lower_bound, 2 * init_stepsize)
        sigma = jnp.maximum(sigma, stepsize_lower_bound)
        return sigma
    
    # Initialize state
    phi0 = jnp.zeros_like(mu)
    psi0 = jnp.zeros_like(nu)
    sigma0 = jnp.asarray(stepsize, dtype=mu.dtype)
    D0 = dual_value(phi0, psi0)
    
    # Initialize arrays for tracking
    errors0 = jnp.full((maxiterations,), jnp.inf, dtype=mu.dtype)
    dual_values0 = jnp.full((maxiterations,), -jnp.inf, dtype=mu.dtype)
    sigmas0 = jnp.zeros((maxiterations,), dtype=mu.dtype)
    
    # Add tracking for H1 relative error metric
    last_grad_sq_phi0 = jnp.zeros((maxiterations,), dtype=mu.dtype)
    last_grad_sq_psi0 = jnp.zeros((maxiterations,), dtype=mu.dtype)
    
    def body(state):
        (i, phi, psi, sigma, D_old, err, errors, dual_values, sigmas,
         last_grad_sq_phi, last_grad_sq_psi) = state
        
        # φ half-step
        rho_phi, _ = pushforward_fn(mu, -psi)
        
        phi, gphi_sq, pde_sol_phi, residual_phi = update_potential(phi, rho_phi, nu, sigma)
        last_grad_sq_phi = last_grad_sq_phi.at[i].set(gphi_sq)
        
        psi = c_transform(phi)
        # maybe not need second ctransform
        phi = c_transform(psi)
        
        # ψ half-step
        rho_psi, _ = pushforward_fn(nu, -phi)
        
        psi, gpsi_sq, pde_sol_psi, residual_psi = update_potential(psi, rho_psi, mu, sigma)
        
        phi = c_transform(psi)
        # maybe not need second ctransform
        psi = c_transform(phi)
        
        D_new = dual_value(phi, psi)
        sigmas = sigmas.at[i].set(sigma)
        
        # Update stepsize (skip first iteration)
        sigma = lax.cond(
            i > 0,
            lambda _: stepsize_update(sigma, D_new, D_old, gpsi_sq),
            lambda _: sigma,
            operand=None
        )
        
        # Compute error based on selected metric
        if error_metric == 'tv_phi':
            rho_nu, _ = pushforward_fn(nu, -phi)
            err = 0.5 * jnp.sum(jnp.abs(rho_nu - mu))
        elif error_metric == 'tv_psi':
            rho_mu, _ = pushforward_fn(mu, -psi)
            err = 0.5 * jnp.sum(jnp.abs(rho_mu - nu))
        elif error_metric == 'l_inf_psi':
            rho_mu, _ = pushforward_fn(mu, -psi)
            err = jnp.max(jnp.abs(rho_mu - nu))
        elif error_metric == 'h1_psi':
            err = gpsi_sq
        elif error_metric == 'h1_psi_relative':
            current_h1 = gpsi_sq
            err = lax.cond(
                i > 0,
                lambda _: jnp.abs(current_h1 - last_grad_sq_psi[i-1]) / jnp.maximum(current_h1, 1e-10),
                lambda _: jnp.inf,
                operand=None
            )
        elif error_metric == 'sqrt_max_grad':
            err = jnp.sqrt(jnp.maximum(gphi_sq, gpsi_sq))
        elif error_metric == 'transportation_cost':
            err = jnp.abs(D_old - D_new)
        elif error_metric == 'transportation_cost_relative':
            err = jnp.abs(D_old - D_new) / jnp.maximum(jnp.abs(D_new), 1e-10)
        else:
            raise ValueError(f"Unknown error_metric: {error_metric}")
        
        last_grad_sq_psi = last_grad_sq_psi.at[i].set(gpsi_sq)
        errors = errors.at[i].set(err)
        dual_values = dual_values.at[i].set(D_new)
        
        if progressbar:
            jax.debug.print(
                "iter {i}: err={e:.3e}, D={D:.6f}, σ={s:.3e}",
                i=i, e=err, D=D_new, s=sigma
            )
        
        return (i + 1, phi, psi, sigma, D_new, err, errors, dual_values, sigmas,
                last_grad_sq_phi, last_grad_sq_psi)
    
    def cond(state):
        i, *_ = state
        # Current error is stored at index i-1
        curr_err = state[6][jnp.maximum(i - 1, 0)]
        if progressbar:
            pass
            # jax.debug.print("cond: i = {}, curr_err = {}", i, curr_err)
        return jnp.logical_and(i < maxiterations, curr_err > tolerance)
    
    # Initial state
    init = (0, phi0, psi0, sigma0, D0, jnp.inf, errors0, dual_values0, sigmas0,
            last_grad_sq_phi0, last_grad_sq_psi0)
    
    # Run optimization loop
    state = lax.while_loop(cond, body, init)
    (iterations, phi, psi, _, _, _, errors, dual_values, sigmas,
     last_grad_sq_phi, last_grad_sq_psi) = state
    
    # Compute final pushforwards for reporting
    rho_mu, _ = pushforward_fn(mu, -psi)  # T_psi # mu (lives on ν-grid)
    rho_nu, _ = pushforward_fn(nu, -phi)  # S_phi # nu (lives on μ-grid)
    
    return (iterations, phi, psi, rho_nu, rho_mu, errors, dual_values, sigmas)
    return (iterations, phi, psi, rho_nu, rho_mu, errors, dual_values)