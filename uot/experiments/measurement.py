import jax
from jax import numpy as jnp
import time
from gpu_tracker.tracker import Tracker
from typing import Any
from uot.utils.instantiate_solver import instantiate_solver


def _wait_jax_finish(result: dict[str, Any]) -> dict[str, Any]:
    """Block until all JAX arrays in `result` are ready."""
    # tree_map was removed and need to use tree.map?
    # ну ебать его в рот, я хуй знает как оно там в джаксе
    return jax.tree.map(
        lambda x: x.block_until_ready() if isinstance(x, jax.Array) else x,
        result
    )


def _require(result: dict[str, Any], required: set[str]) -> None:
    missing = required - result.keys()
    if missing:
        raise RuntimeError(f"Solver returned no `{missing}` fields")


def measure_time(prob, instance, marginals, costs, **kwargs):
    # solver_init_kwargs = kwargs or {}
    # instance = instantiate_solver(solver_cls=solver, init_kwargs=solver_init_kwargs)
    start_time = time.perf_counter()
    solution = instance.solve(marginals=marginals, costs=costs, **kwargs)
    _wait_jax_finish(solution)
    # for metrics we return time in ms units
    metrics = {"time": (time.perf_counter() - start_time) * 1000}
    return metrics

def measure_time_and_output(prob, instance, marginals, costs, **kwargs):
    # instance = solver()
    start_time = time.perf_counter()
    solution = instance.solve(marginals=marginals, costs=costs, **kwargs)
    _wait_jax_finish(solution)
    metrics = {"time": (time.perf_counter() - start_time) * 1000}
    metrics.update(solution)
    return metrics


def measure_solution_precision(prob, instance, *args, **kwargs):
    # solver_init_kwargs = kwargs or {}
    # instance = instantiate_solver(solver_cls=solver, init_kwargs=solver_init_kwargs)
    result = instance.solve(*args, **kwargs)
    _wait_jax_finish(result)
    _require(result, {'cost'})
    metrics = {
        "cost_rerr": abs(prob.get_exact_cost() - result['cost']) / prob.get_exact_cost()
    }
    return metrics


def measure_with_gpu_tracker(prob, instance, *args, **kwargs):
    # solver_init_kwargs = kwargs or {}
    # instance = instantiate_solver(solver_cls=solver, init_kwargs=solver_init_kwargs)
    with Tracker(
        sleep_time=0.1,
        gpu_ram_unit='megabytes',
        time_unit='seconds',
    ) as gt:
        # if learning rate finder is present we firstly estimate the best lr
        # if hasattr(instance, 'find_lr'):
        #     lrs, losses = instance.find_lr(*args, **kwargs)
        #     print("Learning rate finder results:")
        #     for lr, loss in zip(lrs, losses):
        #         print(f"  lr: {lr:.3e}, loss: {loss:.5e}")
        #     # pick lr with minimal loss
        #     min_idx = jnp.argmin(losses)
        #     lr = float(lrs[min_idx])
        #     print(f"Selected learning rate: {lr:.3e}")

        start_time = time.perf_counter()
        metrics = instance.solve(*args, **kwargs)
        _wait_jax_finish(metrics)
        finish_time = time.perf_counter()

        # additinally get some extra metrics for Back-and-Forth solver
        if instance.__class__.__name__ == 'BackNForthSqEuclideanSolver':
            marginals = args[0]
            axes_mu = marginals[0].for_grid_solver(backend="jax", dtype=jnp.float64)[0]
            grids = jnp.meshgrid(*axes_mu, indexing="ij")     # list of d arrays, each (*shape)
            X = jnp.stack(grids, axis=-1)
            (mu_nd, nu_nd) = [
                marginals[0].for_grid_solver(backend="jax", dtype=jnp.float64)[1],
                marginals[1].for_grid_solver(backend="jax", dtype=jnp.float64)[1],
            ]
            extra = instance._extra_metrics(
                mu_nd=mu_nd,
                nu_nd=nu_nd,
                axes_mu=axes_mu,
                X=X,
                psi=-metrics['v_final'],
                T=metrics['monge_map'],
            )
            metrics.update(extra)
            
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

    # don't need the map for these tests
    metrics.pop('monge_map', None)

    _require(metrics, {'cost'})
    metrics.update({
        # NOTE: consider dropping this as we have a separate measurement function
        #       to measure the precision plus run LP solver separately.
        # "cost_rerr": abs(prob.get_exact_cost() - metrics['cost']) / prob.get_exact_cost(),
        'cost': metrics.get('cost', None),

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
