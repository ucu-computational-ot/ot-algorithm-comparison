# import jax
# from jax import lax
# from jax import numpy as jnp
# from jax.scipy.fft import dctn, idctn
# from functools import partial
# from .c_transform import c_transform_quadratic_fast
# from .pushforward import adaptive_pushforward_nd



# # ------------------------ main BFM (d-dimensional) ------------------------
# @partial(jax.jit, static_argnames=('maxiterations', 'progressbar',
#                                   'stepsize_lower_bound', 'error_metric',
#                                   'pushforward_fn'))
# def backnforth_sqeuclidean_nd(
#         mu: jnp.ndarray,                 # shape (n0,...,nd-1)
#         nu: jnp.ndarray,                 # shape (n0,...,nd-1)
#         coordinates: list[jnp.ndarray],  # len d, each length n_k
#         stepsize: float,
#         maxiterations: int,
#         tolerance: float,
#         progressbar: bool = False,
#         pushforward_fn=adaptive_pushforward_nd,
#         stepsize_lower_bound: float = 0.01,
#         error_metric: str = 'h1_psi',
#     ):
#     """
#     Dimension-agnostic BFM with quadratic cost on a uniform tensor grid in [0,1]^d.

#     error_metric: 'tv_psi' | 'tv_phi' | 'l_inf_psi' | 'h1_psi' | 'h1_psi_relative'
#                   | 'transportation_cost' | 'transportation_cost_relative'
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
#     armijo_upper = 0.75
#     armijo_lower = 0.25
#     armijo_scale_down = 0.95

#     # c-transform for quadratic cost (will call your fast implementation)
#     c_transform = partial(c_transform_quadratic_fast, coords_list=coordinates)

#     # precompute kernel and r^2 grid
#     kernel = neumann_kernel_nd(shape, Ls, dtype=mu.dtype)
#     r2 = _r2_from_coords(coordinates)
#     cell_vol = jnp.prod(jnp.array([c[1] - c[0] for c in coordinates], dtype=mu.dtype))
#     mu_nu_grid_sum = 0.5 * (r2 * (mu + nu)).sum()


#     def dct_neumann_poisson(f):
#         f = f - f.mean()
#         Fh = _dctn(f)
#         Uh = Fh / kernel
#         Uh = Uh.at[(0,)*f.ndim].set(0.0)
#         u = _idctn(Uh)
#         return u - u.mean()

#     def update_potential(phi, rho, target, sigma):
#         residual = target - rho
#         pde_solution = dct_neumann_poisson(residual)
#         new_phi = phi + sigma * pde_solution
#         grad_sq = jnp.vdot(residual, pde_solution).real
#         return new_phi, pde_solution, grad_sq

#     # Dual objective (quadratic cost):  ½∫|x|² (μ+ν) - ∫ν φ - ∫μ ψ
#     def dual_value(phi, psi):
#         return mu_nu_grid_sum - (phi * mu).sum() - (psi * nu).sum()

#     # Armijo–Goldstein heuristic
#     def stepsize_update(sigma, value, old_value, grad_sq,
#                         upper=armijo_upper,
#                         lower=armijo_lower,
#                         scale_down=armijo_scale_down):
#         scale_up = 1.0 / scale_down
#         gain = value - old_value
#         old_sigma = sigma
#         sigma = jnp.where(
#             gain > sigma * upper * grad_sq,
#             sigma * scale_up, sigma
#         )
#         sigma = jnp.where(
#             gain < sigma * lower * grad_sq,
#             sigma * scale_down, sigma
#         )
#         if progressbar:
#             jax.debug.print("[stepsize_update] gain = {}; up = {}; low = {}; sigma {} -> {}",
#                             gain, grad_sq * sigma * upper,
#                             grad_sq * sigma * lower, old_sigma, sigma)
#         sigma = jnp.maximum(sigma, stepsize_lower_bound)
#         return sigma

#     def compute_error(iter_idx, phi, psi, dual_curr, dual_prev, grad_curr, grad_prev):
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
#             err = grad_curr
#         elif error_metric == 'h1_psi_relative':
#             err = jnp.where(
#                 iter_idx == 0,
#                 jnp.inf,
#                 jnp.abs(grad_prev - grad_curr) / jnp.maximum(grad_curr, 1e-10),
#             )
#         elif error_metric == 'transportation_cost':
#             err = jnp.abs(dual_prev - dual_curr)
#         elif error_metric == 'transportation_cost_relative':
#             err = jnp.where(
#                 iter_idx == 0,
#                 jnp.inf,
#                 jnp.abs(dual_prev - dual_curr) / jnp.maximum(jnp.abs(dual_curr), 1e-10),
#             )
#         else:
#             raise ValueError(f"Unknown error_metric: {error_metric}")
#         return err
    
#     def half_step(stage, phi, psi, sigma):
#         source, target = (mu, nu) if stage == 'phi' else (nu, mu)
#         active, passive = (phi, psi) if stage == 'phi' else (psi, phi)
#         rho, _ = pushforward_fn(source, -passive)
#         active, _, grad_sq = update_potential(active, rho, target, sigma)
#         if stage == 'phi':
#             phi = active
#             psi = c_transform(phi)
#         else:
#             psi = active
#             phi = c_transform(psi)
#         dual = dual_value(phi, psi)
#         return phi, psi, grad_sq, dual

#     def body(state):
#         (i, phi, psi, sigma, dual_prev, grad_prev,
#          errors, dual_values, sigma_history) = state

#         phi, psi, _, _ = half_step('phi', phi, psi, sigma)
#         phi, psi, grad_sq_psi, dual_curr = half_step('psi', phi, psi, sigma)

#         sigma_new = stepsize_update(sigma, dual_curr, dual_prev, grad_sq_psi)
#         err = compute_error(i, phi, psi, dual_curr, dual_prev, grad_sq_psi, grad_prev)

#         errors = errors.at[i].set(err)
#         dual_values = dual_values.at[i].set(dual_curr)
#         sigma_history = sigma_history.at[i].set(sigma_new)

#         return (i + 1, phi, psi, sigma_new, dual_curr, grad_sq_psi,
#                 errors, dual_values, sigma_history)

#     def cond(state):
#         i = state[0]
#         errors = state[6]
#         curr_error = errors[jnp.maximum(i - 1, 0)]
#         return (i < maxiterations) & (curr_error > tolerance)

#     phi0 = jnp.zeros_like(mu)
#     psi0 = jnp.zeros_like(nu)
#     dual0 = dual_value(phi0, psi0)
#     grad0 = 0.0

#     errors0 = jnp.full((maxiterations,), jnp.inf, dtype=mu.dtype)
#     dual_values0 = jnp.full((maxiterations,), dual0, dtype=mu.dtype)
#     sigma_history0 = jnp.full((maxiterations,), stepsize, dtype=mu.dtype)

#     init_state = (jnp.array(0, dtype=jnp.int32), phi0, psi0,
#                   jnp.asarray(stepsize, dtype=mu.dtype), dual0, grad0,
#                   errors0, dual_values0, sigma_history0)

#     state = lax.while_loop(cond, body, init_state)
#     iterations, phi, psi, _, _, _, errors, dual_values, sigma_history = state

#     rho_mu, _ = pushforward_fn(mu, -psi)
#     rho_nu, _ = pushforward_fn(nu, -phi)
#     return iterations, phi, psi, rho_nu, rho_mu, errors, dual_values, sigma_history


# def _r2_from_coords(coords):
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
#     d = len(shape)
#     hs = [L / N for L, N in zip(lengths, shape)]
#     parts = []

#     for i, (N, h) in enumerate(zip(shape, hs)):
#         k = jnp.arange(N, dtype=dtype)
#         lam1d = (4.0 / (h * h)) * jnp.sin(jnp.pi * k / (2 * N)) ** 2
#         sh = (1,) * i + (N,) + (1,) * (d - i - 1)
#         parts.append(jnp.reshape(lam1d, sh))

#     Lam = jnp.sum(jnp.stack([jnp.broadcast_to(p, shape) for p in parts], axis=0), axis=0).astype(dtype)
#     Lam = Lam.at[(0,) * d].set(jnp.inf)
#     return Lam

import jax
from jax import lax
from jax import numpy as jnp
from jax.scipy.fft import dctn, idctn
import numpy as np
from functools import partial
from .c_transform import c_transform_quadratic_fast
from .pushforward import adaptive_pushforward_nd



# ------------------------ main BFM (d-dimensional) ------------------------
@partial(jax.jit, static_argnames=('maxiterations', 'progressbar',
                                  'stepsize_lower_bound', 'error_metric',
                                  'pushforward_fn'))
def backnforth_sqeuclidean_nd(
        mu: jnp.ndarray,                 # shape (n0,...,nd-1)
        nu: jnp.ndarray,                 # shape (n0,...,nd-1)
        coordinates: list[jnp.ndarray],  # len d, each length n_k
        stepsize: float,
        maxiterations: int,
        tolerance: float,
        progressbar: bool = False,
        pushforward_fn=adaptive_pushforward_nd,
        stepsize_lower_bound: float = 0.01,
        error_metric: str = 'h1_psi',
    ):
    """
    Dimension-agnostic BFM with quadratic cost on a uniform tensor grid in [0,1]^d.

    error_metric: 'tv_psi' | 'tv_phi' | 'l_inf_psi' | 'h1_psi' | 'h1_psi_relative'
                  | 'transportation_cost' | 'transportation_cost_relative'
    """

    # checks (lightweight; keep in Python tracer-friendly)
    shape = mu.shape
    assert nu.shape == shape
    d = len(coordinates)
    assert d == mu.ndim == nu.ndim
    for k in range(d):
        assert coordinates[k].shape[0] == shape[k]
    Ls = [coord[-1] for coord in coordinates]
    init_stepsize = stepsize
    armijo_upper = 0.75
    armijo_lower = 0.25
    armijo_scale_down = 0.95

    # c-transform for quadratic cost (will call your fast implementation)
    c_transform = partial(c_transform_quadratic_fast, coords_list=coordinates)

    # precompute kernel and r^2 grid
    kernel = neumann_kernel_nd(shape, Ls, dtype=mu.dtype)
    r2 = _r2_from_coords(coordinates)
    cell_vol = jnp.prod(jnp.array([c[1] - c[0] for c in coordinates], dtype=mu.dtype))
    mu_nu_grid_sum = 0.5 * (r2 * (mu + nu)).sum()


    def dct_neumann_poisson(f):
        f = f - f.mean()
        Fh = _dctn(f)
        Uh = Fh / kernel
        Uh = Uh.at[(0,)*f.ndim].set(0.0)
        u = _idctn(Uh)
        return u - u.mean()

    def update_potential(phi, rho, target, sigma):
        residual = target - rho
        pde_solution = dct_neumann_poisson(residual)
        new_phi = phi + sigma * pde_solution
        grad_sq = cell_vol * jnp.vdot(residual, pde_solution).real
        return new_phi, pde_solution, grad_sq

    # Dual objective (quadratic cost):  ½∫|x|² (μ+ν) - ∫ν φ - ∫μ ψ
    def dual_value(phi, psi):
        return cell_vol * (mu_nu_grid_sum - (phi * mu).sum() - (psi * nu).sum())

    # Armijo–Goldstein heuristic
    def stepsize_update(sigma, value, old_value, grad_sq,
                        upper=armijo_upper,
                        lower=armijo_lower,
                        scale_down=armijo_scale_down):
        scale_up = 1.0 / scale_down
        gain = value - old_value
        old_sigma = sigma
        sigma = jnp.where(
            gain > sigma * upper * grad_sq,
            sigma * scale_up, sigma
        )
        sigma = jnp.where(
            gain < sigma * lower * grad_sq,
            sigma * scale_down, sigma
        )
        if progressbar:
            jax.debug.print("[stepsize_update] gain = {}; up = {}; low = {}; sigma {} -> {}",
                            gain, grad_sq * sigma * upper,
                            grad_sq * sigma * lower, old_sigma, sigma)
        sigma = jnp.maximum(sigma, stepsize_lower_bound)
        return sigma

    def compute_error(iter_idx, dual_curr, dual_prev, grad_curr, grad_prev,
                      rho_mu=None, rho_nu=None):
        if error_metric == 'tv_psi':
            err = 0.5 * jnp.sum(jnp.abs(rho_mu - nu))
        elif error_metric == 'tv_phi':
            err = 0.5 * jnp.sum(jnp.abs(rho_nu - mu))
        elif error_metric == 'l_inf_psi':
            err = jnp.max(jnp.abs(rho_mu - nu))
        elif error_metric == 'h1_psi':
            err = grad_curr
        elif error_metric == 'h1_psi_relative':
            err = jnp.where(
                iter_idx == 0,
                jnp.inf,
                jnp.abs(grad_prev - grad_curr) / jnp.maximum(grad_curr, 1e-10),
            )
        elif error_metric == 'transportation_cost':
            err = jnp.abs(dual_prev - dual_curr)
        elif error_metric == 'transportation_cost_relative':
            err = jnp.where(
                iter_idx == 0,
                jnp.inf,
                jnp.abs(dual_prev - dual_curr) / jnp.maximum(jnp.abs(dual_curr), 1e-10),
            )
        else:
            raise ValueError(f"Unknown error_metric: {error_metric}")
        return err
    def body(state):
        (i, phi, psi, sigma, dual_prev, grad_prev,
         errors, dual_values, sigma_history) = state

        rho_mu, _ = pushforward_fn(mu, -psi)
        phi, _, _ = update_potential(phi, rho_mu, nu, sigma)
        psi = c_transform(phi)
        phi = c_transform(psi)  # ensure consistency

        rho_nu, _ = pushforward_fn(nu, -phi)
        psi, _, grad_sq_psi = update_potential(psi, rho_nu, mu, sigma)
        phi = c_transform(psi)
        psi = c_transform(phi)  # ensure consistency

        dual_curr = dual_value(phi, psi)
        sigma_new = stepsize_update(sigma, dual_curr, dual_prev, grad_sq_psi)
        err = compute_error(i, dual_curr, dual_prev, grad_sq_psi, grad_prev,
                            rho_mu=rho_mu, rho_nu=rho_nu)

        errors = errors.at[i].set(err)
        dual_values = dual_values.at[i].set(dual_curr)
        sigma_history = sigma_history.at[i].set(sigma_new)

        return (i + 1, phi, psi, sigma_new, dual_curr, grad_sq_psi,
                errors, dual_values, sigma_history)

    def cond(state):
        i = state[0]
        errors = state[6]
        curr_error = errors[jnp.maximum(i - 1, 0)]
        return (i < maxiterations) & (curr_error > tolerance)

    phi0 = jnp.zeros_like(mu)
    psi0 = jnp.zeros_like(nu)
    dual0 = dual_value(phi0, psi0)
    grad0 = 0.0

    errors0 = jnp.full((maxiterations,), jnp.inf, dtype=mu.dtype)
    dual_values0 = jnp.full((maxiterations,), dual0, dtype=mu.dtype)
    sigma_history0 = jnp.full((maxiterations,), stepsize, dtype=mu.dtype)

    init_state = (jnp.array(0, dtype=jnp.int32), phi0, psi0,
                  jnp.asarray(stepsize, dtype=mu.dtype), dual0, grad0,
                  errors0, dual_values0, sigma_history0)

    state = lax.while_loop(cond, body, init_state)
    iterations, phi, psi, _, _, _, errors, dual_values, sigma_history = state

    rho_mu, _ = pushforward_fn(mu, -psi)
    rho_nu, _ = pushforward_fn(nu, -phi)
    return iterations, phi, psi, rho_nu, rho_mu, errors, dual_values, sigma_history


def _r2_from_coords(coords):
    grids = jnp.meshgrid(*coords, indexing="ij")
    r2 = jnp.zeros_like(grids[0])
    for G in grids:
        r2 = r2 + G * G
    return r2


def _dctn(a):
    return dctn(a, type=2, norm="ortho")


def _idctn(a):
    return idctn(a, type=2, norm="ortho")


def neumann_kernel_nd(shape, lengths, dtype=jnp.float64):
    d = len(shape)
    hs = [L / N for L, N in zip(lengths, shape)]
    parts = []

    for i, (N, h) in enumerate(zip(shape, hs)):
        k = jnp.arange(N, dtype=dtype)
        lam1d = (4.0 / (h * h)) * jnp.sin(jnp.pi * k / (2 * N)) ** 2
        sh = (1,) * i + (N,) + (1,) * (d - i - 1)
        parts.append(jnp.reshape(lam1d, sh))

    Lam = jnp.sum(jnp.stack([jnp.broadcast_to(p, shape) for p in parts], axis=0), axis=0).astype(dtype)
    Lam = Lam.at[(0,) * d].set(jnp.inf)
    return Lam
