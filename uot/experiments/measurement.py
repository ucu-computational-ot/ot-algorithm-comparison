import jax
import time
from gpu_tracker import GPUTracker


def measure_time(prob, solver, marginals, costs, **kwargs):
    start_time = time.perf_counter()
    metrics = solver().solve(marginals=marginals, costs=costs, **kwargs)
    if isinstance(metrics['transport_plan'], jax.Array):
        metrics['transport_plan'].block_until_ready()
    metrics["time"] = (time.perf_counter() - start_time) * 1000
    return metrics


def measure_with_tracker(prob, solver, *args, **kwargs):
    with GPUTracker() as gt:
        metrics = solver().solve(*args, **kwargs)
        # for now drop transport plan and potentials
        metrics.pop('transport_plan', None)
        metrics.pop('u_final', None)
        metrics.pop('v_final', None)
    stats = gt.get_stats()
    metrics.update({
        'time': stats['time_sec'] * 1000,
        'peak_mem_MiB': stats['max_gpu_ram_MB'],
        'mean_mem_MiB': stats['mean_gpu_ram_MB'],
        'peak_util_pct': stats['max_gpu_util_pct'],
        'mean_util_pct': stats['mean_gpu_util_pct'],
    })
    return metrics
