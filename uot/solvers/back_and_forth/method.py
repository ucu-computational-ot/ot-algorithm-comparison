import jax
from jax import lax
from jax import numpy as jnp
from jax.scipy.fft import dctn, idctn
from functools import partial
from .c_transform import c_transform_quadratic_fast
from .pushforward import _forward_pushforward_nd


# ------------------------ main BFM (d-dimensional) ------------------------
@partial(jax.jit, static_argnames=('maxiterations', 'progressbar', 'stepsize_lower_bound', 'error_metric'))
def backnforth_sqeuclidean_nd(
        mu: jnp.ndarray,                 # shape (n0,...,nd-1)
        nu: jnp.ndarray,                 # shape (n0,...,nd-1)
        coordinates: list[jnp.ndarray],  # len d, each length n_k
        stepsize: float,
        maxiterations: int,
        tolerance: float,
        progressbar: bool = False,
        pushforward_fn=_forward_pushforward_nd,
        stepsize_lower_bound: float = 0.01,
        error_metric: str = 'h1_psi',
    ):
    """
    Dimension-agnostic BFM with quadratic cost on a uniform tensor grid in [0,1]^d.

    error_metric: 'tv_psi' | 'tv_phi' | 'l_inf_psi' | 'h1_psi' | 'h1_psi_relative' | 'transportation_cost' | 'transportation_cost_relative'
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

    # c-transform for quadratic cost (will call your fast implementation)
    c_transform = partial(c_transform_quadratic_fast, coords_list=coordinates)

    # precompute kernel and r^2 grid
    kernel = neumann_kernel_nd(shape, Ls, dtype=mu.dtype)
    r2 = _r2_from_coords(coordinates)
    cell_vol = jnp.prod(jnp.array([c[1] - c[0] for c in coordinates], dtype=mu.dtype))

    def dct_neumann_poisson(f):
        """
        Solve -Δu = f with Neumann BC via DCT-II (up to constant)
        on cell-centered grid.
        """
        f = f - f.mean()
        Fh = _dctn(f)
        Uh = Fh / kernel
        Uh = Uh.at[(0,)*f.ndim].set(0.0)    # DC null mode
        u = _idctn(Uh)
        return u - u.mean()

    # φ ← φ + σ Δ^{-1}(ρ−ν), return ⟨(-Δ)^{-1}(ρ−ν), (ρ−ν)⟩
    def update_potential(phi, rho, target, sigma):
        rho_diff = target - rho
        recov = dct_neumann_poisson(rho_diff)
        new_phi = phi + sigma * recov
        grad_sq = cell_vol * jnp.vdot(rho_diff, recov).real
        return new_phi, recov, grad_sq

    # Dual objective (quadratic cost):  ½∫|x|² (μ+ν) - ∫ν φ - ∫μ ψ
    def dual_value(phi, psi, mu, nu):
        return cell_vol * jnp.sum(0.5 * r2 * (mu + nu) - nu * phi - mu * psi)

    # Armijo–Goldstein heuristic
    def stepsize_update(sigma, value, old_value, grad_sq, upper=0.75, lower=0.25, scale_down=0.985):
        scale_up = 1.0 / scale_down
        gain = value - old_value
        sigma = jnp.where(
            gain > sigma * upper * grad_sq,
            sigma * scale_down, sigma
        )
        sigma = jnp.where(
            gain < sigma * lower * grad_sq,
            sigma * scale_up, sigma
        )
        sigma = jnp.clip(sigma, stepsize_lower_bound, 2 * init_stepsize)
        return sigma

    def body(state):
        (i, phi, psi, D_value_old, sigma, last_gradient_seminorms, errors, dual_values) = state

        # --- φ half-step ---
        # pushforward (ψ acts on μ)
        rho_phi, _ = pushforward_fn(mu, -psi)
        phi, pde_sol_phi, grad_sq_phi = update_potential(phi, rho_phi, nu, sigma)
        psi = c_transform(phi)
        phi = c_transform(psi)
        D_value = dual_value(phi, psi, mu, nu)
        # update stepsize after update on J(φ)
        sigma = lax.cond(i > 0,
                         lambda _: stepsize_update(
                            sigma, D_value, D_value_old, last_gradient_seminorms['J']),
                         lambda _: sigma,
                         operand=None)
        D_value_old = D_value
        last_gradient_seminorms['J'] = grad_sq_phi

        # --- ψ half-step ---
        # pushforward (φ acts on ν)
        rho_psi, _ = pushforward_fn(nu, -phi)
        psi, pde_sol_psi, grad_sq_psi = update_potential(psi, rho_psi, mu, sigma)
        phi = c_transform(psi)
        psi = c_transform(phi)
        D_value = dual_value(phi, psi, mu, nu)
        # update stepsize after update on I(ψ)
        sigma = lax.cond(i > 0,
                         lambda _: stepsize_update(
                            sigma, D_value, D_value_old, last_gradient_seminorms['I']),
                         lambda _: sigma,
                         operand=None)

        # parametrized error computation (static branching->one path compiled)
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
            err = grad_sq_psi
        elif error_metric == 'h1_psi_relative':
            err = jnp.abs(
                last_gradient_seminorms['I'] - grad_sq_psi) / jnp.maximum(
                grad_sq_psi, 1e-10)
        elif error_metric == 'transportation_cost':
            err = jnp.abs(D_value_old - D_value)
        elif error_metric == 'transportation_cost_relative':
            err = jnp.abs(D_value_old - D_value) / jnp.maximum(
                jnp.abs(D_value), 1e-10)
        else:
            raise ValueError(f"Unknown error_metric: {error_metric}")
        errors = errors.at[i].set(err)

        last_gradient_seminorms['I'] = grad_sq_psi
        dual_values = dual_values.at[i].set(D_value)
        return (i+1, phi, psi, D_value, sigma, last_gradient_seminorms, errors, dual_values)

    def cond(state):
        i = state[0]
        curr_error = state[6][jnp.maximum(i - 1, 0)]
        return (i < maxiterations) & (curr_error > tolerance)

    # init
    phi0 = jnp.zeros_like(mu)
    psi0 = jnp.zeros_like(nu)
    dual0 = dual_value(phi0, psi0, mu, nu)
    gradient_seminorms0 = {'J': -jnp.inf, 'I': jnp.inf}
    errors0 = jnp.full((maxiterations,), jnp.inf, dtype=mu.dtype)
    dual_values0 = jnp.full((maxiterations,), -jnp.inf, dtype=mu.dtype)
    init = (
        0, phi0, psi0, dual0, stepsize,
        gradient_seminorms0, errors0, dual_values0)

    state = lax.while_loop(cond, body, init)
    iterations, phi, psi, _, _, _, errors, dual_values = state

    rho_mu, _ = pushforward_fn(mu, -psi)
    rho_nu, _ = pushforward_fn(nu, -phi)
    return iterations, -phi, -psi, rho_nu, rho_mu, errors, dual_values


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


def _dctn(a):
    return dctn(a, type=2, norm="ortho")


def _idctn(a):
    return idctn(a, type=2, norm="ortho")


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
