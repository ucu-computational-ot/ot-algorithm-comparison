import jax
import time
from gpu_tracker.tracker import Tracker
from typing import Any
from uot.utils.instantiate_solver import instantiate_solver
import jax.numpy as jnp
import ot
from ucp.benchmarks.cudf_merge import exchange_and_concat_bins


def _wait_jax_finish(result: dict[str, Any]) -> dict[str, Any]:
    """Block until all JAX arrays in `result` are ready."""
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if isinstance(x, jax.Array) else x,
        result
    )


def _require(result: dict[str, Any], required: set[str]) -> None:
    missing = required - result.keys()
    if missing:
        raise RuntimeError(f"Solver returned no `{missing}` fields")


def measure_time(prob, solver, marginals, costs, **kwargs):
    solver_init_kwargs = kwargs or {}
    instance = instantiate_solver(solver_cls=solver, init_kwargs=solver_init_kwargs)
    start_time = time.perf_counter()
    solution = instance.solve(marginals=marginals, costs=costs, **kwargs)
    _wait_jax_finish(solution)
    # for metrics we return time in ms units
    metrics = {"time": (time.perf_counter() - start_time) * 1000}
    return metrics


def measure_solution_precision(prob, solver, *args, **kwargs):
    solver_init_kwargs = kwargs or {}
    instance = instantiate_solver(solver_cls=solver, init_kwargs=solver_init_kwargs)
    result = instance.solve(*args, **kwargs)
    _wait_jax_finish(result)
    _require(result, {'cost'})
    metrics = {
        "cost_err": abs(prob.get_exact_cost() - result['cost']) / prob.get_exact_cost()
    }
    return metrics

def measure_time_and_precision(prob, solver, *args, **kwargs):
    instance = solver()
    start_time = time.perf_counter()
    result = instance.solve(*args, **kwargs)
    _wait_jax_finish(result)
    end_time = (time.perf_counter() - start_time) * 1000
    _require(result, {'cost'})
    exact_cost = prob.get_exact_cost()
    metrics = {
        "time": end_time,
        "cost_err": abs(exact_cost - result['cost']) / exact_cost
    }
    return metrics

def measure_with_gpu_tracker(prob, solver, *args, **kwargs):
    solver_init_kwargs = kwargs or {}
    instance = instantiate_solver(solver_cls=solver, init_kwargs=solver_init_kwargs)
    with Tracker(
        sleep_time=0.1,
        gpu_ram_unit='megabytes',
        time_unit='seconds',
    ) as gt:
        start_time = time.perf_counter()
        metrics = instance.solve(*args, **kwargs)
        _wait_jax_finish(metrics)
        finish_time = time.perf_counter()
        # save some other metrics but drop plan and potintials
        # as they use too much memory
        metrics.pop('transport_plan', None)
        metrics.pop('u_final', None)
        metrics.pop('v_final', None)

    usage = gt.resource_usage
    peak_gpu_ram = usage.max_gpu_ram
    gpu_utilization = usage.gpu_utilization
    peak_ram = usage.max_ram
    cpu_utilization = usage.cpu_utilization
    time_counter = finish_time - start_time
    _require(metrics, {'cost'})
    metrics.update({
        "cost_err": abs(prob.get_exact_cost() - metrics['cost']) / prob.get_exact_cost(),

        'time_unit': usage.compute_time.unit,
        'time': usage.compute_time.time,
        'time_counter': time_counter,
        # GPU MEMORY
        'gpu_mem_unit':              peak_gpu_ram.unit,
        'peak_gpu_mem':              peak_gpu_ram.main,
        'combined_peak_gpu_ram':     peak_gpu_ram.combined,
        # GPU UTILIZATION
        'peak_gpu_util_pct': gpu_utilization.gpu_percentages.max_hardware_percent,
        'mean_gpu_util_pct': gpu_utilization.gpu_percentages.mean_hardware_percent,
        # MEMORY
        "peak_ram_MiB":              peak_ram.main.private_rss,
        "combined_peak_ram_MiB":     peak_ram.combined.private_rss,
        # CPU UTILIZATION
        "max_cpu_util_pct":          cpu_utilization.main.max_hardware_percent,
        "mean_cpu_util_pct":         cpu_utilization.main.mean_hardware_percent,
    })
    return metrics

# ---------- helpers -----------------------------------------------------------
# ────────────── utilities ───────────────────────────────────────────
_vec = lambda x: jnp.asarray(x).reshape(-1)          # force 1‑D

def get_marginal_params(marginal):
    means = marginal.means[0] if marginal.means.shape == (1, 1) else marginal.means
    covs = jnp.sqrt(marginal.covs[0][0]) if marginal.covs.shape == (1, 1, 1) else jnp.sqrt(marginal.covs)
    weights = marginal.comp_weights
    return means, covs, weights

def gm_pdf(x, means, sigmas, weights=None):
    means  = jnp.asarray(means,  dtype=float)
    sigmas = jnp.asarray(sigmas, dtype=float)
    if weights is None:
        weights = jnp.ones_like(means) / means.size
    else:
        weights = jnp.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    x = jnp.asarray(x, dtype=float)[..., None]          # broadcast over comps
    z = (x - means) / sigmas
    comp_pdf = jnp.exp(-0.5 * z**2) / (sigmas * jnp.sqrt(2.0 * jnp.pi))
    return jnp.sum(weights * comp_pdf, axis=-1)

def _cost_1d(x):                                     # pair‑wise ‖x−y‖² on a grid
    xx = x[:, None]
    return ot.dist(xx, xx) ** 2

# ────────────── barycentric branch ──────────────────────────────────
def bary_proj(pi, y, a):                             # T_i = Σ_j π_ij y_j / a_i
    y, a = _vec(y), _vec(a)
    T = (pi @ y) / (a + 1e-12)
    return T

def rebin(T, w, grid):                               # push mass back to grid
    T, w, grid = _vec(T), _vec(w), _vec(grid)
    idx  = jnp.argmin(jnp.abs(T[:, None] - grid[None, :]), axis=1)
    return jnp.zeros_like(grid).at[idx].add(w)

def bary_bias(mu_w, nu_x, nu_w, pi):
    T         = bary_proj(pi, nu_x, mu_w)
    nu_hat_w  = rebin(T, mu_w, nu_x)
    l2   = jnp.linalg.norm(nu_hat_w - nu_w)
    w2   = jnp.sqrt(ot.emd2(nu_hat_w, nu_w, _cost_1d(nu_x)))
    return l2, w2

# ────────────── Monge‑map branch (interp + AD) ──────────────────────
def interp_scalar(x, grid, u):
    grid, u = _vec(grid), _vec(u)
    i  = jnp.clip(jnp.searchsorted(grid, x) - 1, 0, grid.size - 2)
    x0, x1 = grid[i], grid[i + 1]
    t  = (x - x0) / (x1 - x0 + 1e-12)
    return ((1 - t) * u[i] + t * u[i + 1]).reshape(())

def monge_bias(mu_x, nu_w, u_vals, nu_params):
    mu_x = _vec(mu_x)
    grad_fn = jax.grad(lambda z: interp_scalar(z, mu_x, u_vals))
    grad = jax.vmap(grad_fn)(mu_x)

    T    = mu_x - grad
    nu_hat_w = gm_pdf(T, *nu_params)
    nu_hat_w = nu_hat_w / nu_hat_w.sum()

    l2   = jnp.linalg.norm(nu_hat_w - nu_w)

    C = (T[:, None] - T[None, :]) ** 2
    w2   = jnp.sqrt(ot.emd2(nu_hat_w, nu_w, C))
    return l2, w2

# ────────────── public entry point ──────────────────────────────────
def measure_pushforward(prob, Solver, *args, **kw):
    res   = Solver().solve(*args, **kw)
    π, u  = res["transport_plan"], _vec(res["u_final"])

    margs = prob.get_marginals()

    (μ_x, μ_w), (ν_x, ν_w) = [m.to_discrete() for m in margs]
    μ_x, μ_w, ν_x, ν_w = map(_vec, (μ_x, μ_w, ν_x, ν_w))

    bl2, bw2 = bary_bias(μ_w, ν_x, ν_w, π)
    ml2, mw2 = monge_bias(μ_x, ν_w, u, get_marginal_params(margs[1]))

    return dict(
        barycentric_l2_bias = bl2,
        barycentric_w2_bias = bw2,
        monge_l2_bias       = ml2,
        monge_w2_bias       = mw2,
    )