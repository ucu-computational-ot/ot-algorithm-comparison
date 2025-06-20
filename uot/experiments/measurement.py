import jax
import time
from gpu_tracker.tracker import Tracker


def measure_time(prob, solver, marginals, costs, **kwargs):
    start_time = time.perf_counter()
    metrics = solver().solve(marginals=marginals, costs=costs, **kwargs)
    if isinstance(metrics['transport_plan'], jax.Array):
        metrics['transport_plan'].block_until_ready()
    metrics["time"] = (time.perf_counter() - start_time) * 1000
    return metrics


def measure_with_gpu_tracker(prob, solver, *args, **kwargs):
    with Tracker(
        sleep_time=0.001,
        gpu_ram_unit='megabytes',
        time_unit='seconds',
    ) as gt:
        metrics = solver().solve(*args, **kwargs)
        # for now drop transport plan and potentials
        metrics.pop('transport_plan', None)
        metrics.pop('u_final', None)
        metrics.pop('v_final', None)

    usage = gt.resource_usage
    peak_gpu_ram = usage.max_gpu_ram
    gpu_utilization = usage.gpu_utilization
    metrics.update({
        'time': usage.compute_time,
        'gpu_mem_unit': peak_gpu_ram.unit,
        'peak_gpu_mem': peak_gpu_ram.main,
        'combined_peak_gpu_ram': peak_gpu_ram.combined,
        'peak_util_pct': gpu_utilization.gpu_percentages.max_hardware_percent,
        'mean_util_pct': gpu_utilization.gpu_percentages.mean_hardware_percent,
    })
    return metrics
