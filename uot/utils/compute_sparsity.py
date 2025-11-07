import jax.numpy as jnp

def compute_sparsity(plan: jnp.ndarray, threshold: float = 1e-10) -> float:
    """
    Compute sparsity percentage of the transport plan.
    
    Args:
        plan: Transport plan matrix (n x m).
        threshold: Entries <= threshold are considered zero.
    
    Returns:
        Sparsity as percentage (0-100).
    """
    non_zero_count = jnp.sum(plan > threshold)
    total_elements = plan.size
    sparsity = (1 - non_zero_count / total_elements) * 100
    return sparsity