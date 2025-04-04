import jax
import jax.numpy as jnp
import tqdm
import optax

# Normal gradient
def gradient_vanilla_logsumexp(
    C, a, b, eps=0.01, alpha=0.1, tolerance=1e-4, max_iterations=1_000
):
    """
    Vanilla gradient ascent for entropy-regularized optimal transport (Kantorovich problem).

    Arguments:
    C -- cost matrix (n x m)
    a -- source distribution (n,)
    b -- target distribution (m,)
    eps -- entropy regularization coefficient
    alpha -- initial learning rate
    tolerance -- stopping criterion for convergence
    max_iterations -- maximum number of iterations

    Returns:
    phi, psi -- dual potentials
    plan -- optimal transport plan
    """

    n, m = C.shape
    phi, psi = jnp.zeros(n), jnp.zeros(m)  # Dual variables initialization

    def compute_gradients(phi, psi):
        """Compute gradients using stabilized softmax approach."""
        log_K = (phi[:, None] + psi[None, :] - C) / eps  # Log of kernel
        log_K_max = jnp.max(log_K, axis=1, keepdims=True)  # Stabilization factor
        K_stable = jnp.exp(log_K - log_K_max)  # Stable exponentiation
        marginals = jnp.exp(log_K_max).flatten() * jnp.sum(
            K_stable, axis=1
        )  # Sum over columns
        grad_phi = a - marginals  # Compute gradient for phi

        log_K_max = jnp.max(log_K, axis=0, keepdims=True)
        K_stable = jnp.exp(log_K - log_K_max)
        marginals = jnp.exp(log_K_max).flatten() * jnp.sum(
            K_stable, axis=0
        )  # Sum over rows
        grad_psi = b - marginals  # Compute gradient for psi

        return grad_phi, grad_psi

    pbar = tqdm(range(max_iterations), desc="OT Vanilla Gradient Ascent", unit="iter")

    for iteration in pbar:
        grad_phi, grad_psi = compute_gradients(phi, psi)

        # Adaptive step size decay
        # step_size = alpha / (1 + 0.1 * iteration)
        step_size = alpha

        # Update dual variables
        phi_new = phi + step_size * grad_phi
        psi_new = psi + step_size * grad_psi

        # Convergence check
        # if (
        #     jnp.linalg.norm(grad_phi) < tolerance
        #     and jnp.linalg.norm(grad_psi) < tolerance
        # ):
        #     break
        if (
            jnp.linalg.norm(phi_new - phi) < tolerance
            and jnp.linalg.norm(psi_new - psi) < tolerance
        ):
            break
        phi = phi_new
        psi = psi_new

    phi = phi_new
    psi = psi_new

    # Compute final transport plan
    log_K = (phi[:, None] + psi[None, :] - C) / eps
    plan = jnp.exp(log_K) * (
        a[:, None] / jnp.sum(jnp.exp(log_K), axis=1, keepdims=True)
    )  # Normalized

    return phi, psi, plan

##############################################################################################

# Gradient using optax
def gradient_ascent_opt(
    C,
    marginals,
    eps=1e-3,
    optimizer=optax.sgd,
    max_iterations=100_000,
    tol=1e-6,
) -> jax.Array:
    N = marginals.shape[0]
    n = marginals.shape[1]

    shapes = [tuple(n if j == i else 1 for j in range(N)) for i in range(N)]
    potentials = jnp.zeros_like(marginals)
    opt_state = optimizer.init(potentials)

    @jax.jit
    def objective(potentials: jax.Array):
        """Computes the dual objective with logsumexp stabilization."""
        potentials_reshaped = [potentials[i].reshape(shapes[i]) for i in range(N)]
        potentials_sum = sum(potentials_reshaped)
        log_sub_entropy = (potentials_sum - C) / eps
        max_log_sub_entropy = jnp.max(log_sub_entropy, axis=0, keepdims=True)
        stable_sum = jnp.exp(max_log_sub_entropy) * jnp.sum(
            jnp.exp(log_sub_entropy - max_log_sub_entropy), axis=0
        )
        dual = potentials * marginals
        return jnp.sum(dual - eps * stable_sum)

    @jax.jit
    def step(state: tuple[int, jax.Array, optax.OptState, float, bool]):
        """Performs one gradient ascent step."""
        i, potentials, opt_state, prev_loss, _ = state
        loss, grad = jax.value_and_grad(objective)(potentials)
        # minus gradient because we are performing gradient ascent
        updates, opt_state = optimizer.update(-grad, opt_state, potentials)
        potentials = optax.apply_updates(potentials, updates)
        # L-infinity norm
        max_change = jnp.max(jnp.abs(potentials - state[0]))
        has_converged = max_change < tol
        return i + 1, potentials, opt_state, loss, has_converged

    def cond_fn(state: tuple[int, jax.Array, optax.OptState, float, bool]):
        i, _, _, _, has_converged = state
        return jnp.logical_and(i < max_iterations, jnp.logical_not(has_converged))

    final_state = jax.lax.while_loop(
        cond_fn, step, (0, potentials, opt_state, jnp.inf, False)
    )

    steps, final_potentials, _, final_loss, _ = final_state
    print(f"DualOT Gradient Ascent has converged after {steps} steps")
    return final_potentials
