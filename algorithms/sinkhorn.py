import jax
import jax.numpy as jnp
from ott.solvers import linear
from ott.geometry import geometry
from ott.solvers.linear import sinkhorn

def sinkhorn(mu, nu, C, epsilon=0.001, niter=1000):

    jax.config.update("jax_enable_x64", True)

    ln_K = -C / epsilon
    ln_mu = jnp.log(mu)
    ln_nu = jnp.log(nu)
    ln_u = jnp.ones_like(ln_mu)
    ln_v = jnp.ones_like(ln_nu)

    def body(i, ln_uv):
        ln_u, ln_v = ln_uv
        ln_u = ln_mu - jax.scipy.special.logsumexp(ln_K + ln_v[None, :], axis=1)
        ln_v = ln_nu - jax.scipy.special.logsumexp(ln_K.T + ln_u[None, :], axis=1)
        return ln_u, ln_v

    ln_u, ln_v = jax.lax.fori_loop(0, niter, body, (ln_u, ln_v))

    transport_plan = jnp.exp(ln_u[:, None] + ln_K + ln_v[None, :])
    cost = jnp.sum(transport_plan * C).item()
    return transport_plan, cost


def ott_jax_sinkhorn(mu, nu, C, epsilon=0.001, niter=1000):
    jax.config.update("jax_enable_x64", True)
    solution = linear.solve(
                geometry.Geometry(cost_matrix=C, epsilon=epsilon),
                lse_mode=True,
                a=mu,
                b=nu,
            )
    
    return solution.matrix, solution.reg_ot_cost