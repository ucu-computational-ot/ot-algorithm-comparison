import jax
from jax import lax
from jax import numpy as jnp
from jax.scipy.fft import dctn, idctn
from functools import partial
from .c_transform import c_transform_quadratic_fast


# ------------------------ main BFM (d-dimensional) ------------------------
@partial(jax.jit, static_argnames=('maxiterations', 'progressbar'))
def backnforth_sqeuclidean_nd(
        mu: jnp.ndarray,                 # shape (n0,...,nd-1)
        nu: jnp.ndarray,                 # shape (n0,...,nd-1)
        coordinates: list[jnp.ndarray],  # len d, each length n_k
        stepsize: float,
        maxiterations: int,
        tolerance: float,
        progressbar: bool = False,
        pushforward_fn=_cic_pushforward_nd,  # allow swapping deposition schemes
    ):
    """
    Dimension-agnostic BFM with quadratic cost on a uniform tensor grid in [0,1]^d.
    """

    # checks (lightweight; keep in Python tracer-friendly)
    shape = mu.shape
    assert nu.shape == shape
    d = len(coordinates)
    assert d == mu.ndim == nu.ndim
    for k in range(d):
        assert coordinates[k].shape[0] == shape[k]

    # c-transform for quadratic cost (will call your fast implementation)
    c_transform = partial(c_transform_quadratic_fast, coords_list=coordinates)

    # precompute kernel and r^2 grid
    kernel = _initialize_kernel_nd(shape)     # Poisson eigenvalues for DCT solver
    r2 = _r2_from_coords(coordinates)

    def dct_poisson(rhs):
        # Solve Δu = rhs with Neumann BC via DCT (up to constant).
        R = _dctn(rhs) / kernel
        R = R.at[(0,) * d].set(0.0)           # zero-mean fix
        return _idctn(R)

    # φ ← φ + σ Δ^{-1}(ρ−ν), return ⟨(-Δ)^{-1}(ρ−ν), (ρ−ν)⟩
    def update_potential(phi, rho, target, sigma):
        rho_diff = target - rho
        recov = dct_poisson(rho_diff)
        new_phi = phi + sigma * recov
        gradSq = jnp.sum(recov * rho_diff) / rho_diff.size
        return new_phi, gradSq

    # Dual objective (quadratic cost):  ½∫|x|² (μ+ν) - ∫ν φ - ∫μ ψ
    def compute_w2(phi, psi, mu, nu):
        return jnp.sum(0.5 * r2 * (mu + nu) - nu * phi - mu * psi) / mu.size

    # Armijo–Goldstein heuristic
    def stepsize_update(sigma, value, oldValue, gradSq, upper=0.75, lower=0.25, scaleDown=0.95):
        scaleUp = 1.0 / scaleDown
        diff = value - oldValue
        sigma = jnp.where(diff > gradSq * sigma * upper, sigma * scaleUp, sigma)
        sigma = jnp.where(diff < gradSq * sigma * lower, sigma * scaleDown, sigma)
        return sigma

    # loop state: i, phi, psi, rho, old_dual_value, sigma, errors, dual_values
    def body(state):
        i, phi, psi, rho, old_dual_value, sigma, errors, dual_values = state

        # φ-step
        phi, grad_sq = update_potential(phi, rho, nu, sigma)
        psi = c_transform(phi)
        value = compute_w2(phi, psi, mu, nu)
        sigma = stepsize_update(sigma, value, old_dual_value, grad_sq)
        old_dual_value = value

        # pushforward (ψ acts on μ)
        rho = pushforward_fn(mu, psi)

        # ψ-step
        psi, grad_sq = update_potential(psi, rho, mu, sigma)
        phi = c_transform(psi)
        value = compute_w2(phi, psi, mu, nu)
        sigma = stepsize_update(sigma, value, old_dual_value, grad_sq)
        old_dual_value = value

        # pushforward (φ acts on ν)
        rho = pushforward_fn(nu, phi)

        errors = errors.at[i].set(grad_sq)
        dual_values = dual_values.at[i].set(value)
        return (i+1, phi, psi, rho, old_dual_value, sigma, errors, dual_values)

    def cond(state):
        i, *_, errors, _ = state
        return (i < maxiterations) & (errors[jnp.maximum(0, i-1)] > tolerance)

    # init
    phi0 = jnp.zeros_like(mu)
    psi0 = jnp.zeros_like(nu)
    rho0 = mu
    dual0 = compute_w2(phi0, psi0, mu, nu)
    errors0 = jnp.full((maxiterations,), jnp.inf, dtype=mu.dtype)
    dual_values0 = jnp.full((maxiterations,), -jnp.inf, dtype=mu.dtype)
    init = (0, phi0, psi0, rho0, dual0, stepsize, errors0, dual_values0)

    state = lax.while_loop(cond, body, init)
    iterations, phi, psi, _, _, _, errors, dual_values = state

    rho_mu = pushforward_fn(mu, psi)
    rho_nu = pushforward_fn(nu, phi)
    return iterations, phi, psi, rho_nu, rho_mu, errors, dual_values


def _r2_from_coords(coords):
    """
    coords: list of d arrays with lengths n_k, assumed uniform per axis.
    Returns r2 grid with shape (n0,...,nd-1), computed as sum_i x_i^2.
    """
    grids = jnp.meshgrid(*coords, indexing="ij")
    r2 = jnp.zeros_like(grids[0])
    for G in grids:
        r2 = r2 + G * G
    return r2


def _initialize_kernel_nd(shape):
    """
    DCT-II / Neumann Poisson kernel eigenvalues:
    λ(k) = Σ_i 2 * n_i^2 * (1 - cos(π * m_i / n_i)), m_i = 0..n_i-1.
    shape: tuple (n0,...,nd-1)
    """
    freqs = [jnp.linspace(0.0, jnp.pi, n, endpoint=False) for n in shape]
    F = jnp.meshgrid(*freqs, indexing="ij")
    kernel = jnp.zeros(shape)
    for ax, n in enumerate(shape):
        kernel = kernel + 2.0 * (n ** 2) * (1.0 - jnp.cos(F[ax]))
    # avoid division by zero at DC
    return kernel.at[(0,) * len(shape)].set(1.0)


def _dctn(a):
    return dctn(a, type=2, norm="ortho")


def _idctn(a):
    # IDCT for DCT-II with 'ortho' is DCT-III; jax.scipy.fft.idctn handles it.
    return idctn(a, type=2, norm="ortho")
