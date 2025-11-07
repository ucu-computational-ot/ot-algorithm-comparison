import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import numpy as np
import jax.numpy as jnp
import time
import numpy as np

from .bfm import backnforth_sqeuclidean_nd
from uot.solvers.back_and_forth.method import backnforth_sqeuclidean_nd as bfm_uot
from uot.solvers.back_and_forth.monge_map import monge_map_from_psi_nd
from uot.solvers.back_and_forth.pushforward import _central_gradient_nd


def _grid_from_axes(axes):
    X0, X1 = np.meshgrid(np.asarray(axes[0]), np.asarray(axes[1]), indexing="ij")
    return X0, X1

def _spacings_from_axes(axes):
    h0 = float(axes[0][1] - axes[0][0])
    h1 = float(axes[1][1] - axes[1][0])
    return h0, h1

def _central_gradient_2d(U, h0, h1):
    """Central differences with one-sided boundaries; U shape (H,W)."""
    U = np.asarray(U)
    H, W = U.shape
    gx = np.zeros_like(U)
    gy = np.zeros_like(U)

    gx[1:-1, :] = (U[2:, :] - U[:-2, :]) / (2.0 * h0)
    gy[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2.0 * h1)

    gx[0,  :] = (U[1,  :] - U[0,  :]) / h0
    gx[-1, :] = (U[-1, :] - U[-2, :]) / h0
    gy[:, 0 ] = (U[:, 1 ] - U[:, 0  ]) / h1
    gy[:, -1] = (U[:, -1] - U[:, -2 ]) / h1

    return np.stack([gx, gy], axis=-1)  # (H,W,2)

def _downsample_quiver(X0, X1, U0, U1, step):
    return X0[::step, ::step], X1[::step, ::step], U0[::step, ::step], U1[::step, ::step]


def make_bfm_animation(
    results,
    *,
    interval=120,          # ms between frames
    save_path=None,        # e.g. "bfm.mp4" or "bfm.gif" (gif uses Pillow)
    show=True,
    stride=1,              # downsample frames spatially: 2 keeps every 2nd pixel
    use_global_norm=True,  # fixed color limits across time (recommended)
    dpi=110,
    cmap="viridis",
    compare=None,          # tuple (phi_explicit, psi_explicit) or None
):
    """
    Build a 2×5 (or 2×6 if compare provided) animation over iterations from `results`:
      results = (
        iterations, phi, psi, rho_nu, rho_mu, errors, dual_vals,
        phi_after_update,    psi_after_update,
        phi_after_ctransform,psi_after_ctransform,
        rho_before_phi_update, rho_before_psi_update,
        pde_sols_phi, pde_sols_psi,
        residuals_phi, residuals_psi, sigmas
      )

    Panels (top row):
      ρ before φ,  (ν − T_ψ μ),  (-Δ)^{-1}(ν − T_ψ μ),  φ after update,  ψ after c-transform,  [φ* if compare]
    Panels (bottom row):
      ρ before ψ,  (μ − T_φ ν),  (-Δ)^{-1}(μ − T_φ ν),  ψ after update,  φ after c-transform,  [ψ* if compare]
    """
    (iterations, phi, psi, rho_nu, rho_mu, errors, dual_vals,
     phi_after_update, psi_after_update,
     phi_after_ctransform, psi_after_ctransform,
     rho_before_phi_update, rho_before_psi_update,
     pde_sols_phi, pde_sols_psi,
     residuals_phi, residuals_psi, sigmas) = results

    iters = int(iterations)

    # Pack panels in drawing order
    top_panels = [
        rho_before_phi_update,    # (t, H, W)
        residuals_phi,            # (t, H, W)
        pde_sols_phi,             # (t, H, W)
        phi_after_update,         # (t, H, W)
        psi_after_ctransform,     # (t, H, W)
    ]
    bot_panels = [
        rho_before_psi_update,    # (t, H, W)
        residuals_psi,            # (t, H, W)
        pde_sols_psi,             # (t, H, W)
        psi_after_update,         # (t, H, W)
        phi_after_ctransform,     # (t, H, W)
    ]
    titles_top = [
        r'$\rho$ before $\phi$ update',
        r'$\nu - T_{\psi}\mu$',
        r'$(-\Delta)^{-1}(\nu - T_{\psi}\mu)$',
        r'$\phi$ after update (for $\nu$)',
        r'$\psi$ after c-transform (for $\mu$)',
    ]
    titles_bot = [
        r'$\rho$ before $\psi$ update',
        r'$\mu - T_{\phi}\nu$',
        r'$(-\Delta)^{-1}(\mu - T_{\phi}\nu)$',
        r'$\psi$ after update (for $\mu$)',
        r'$\phi$ after c-transform (for $\nu$)',
    ]

    # Optional comparison (explicit φ*, ψ*) as a 6th column
    have_compare = compare is not None
    if have_compare:
        phi_star, psi_star = compare
        # Ensure np arrays and optional downsampling
        phi_star = np.asarray(phi_star)[::stride, ::stride]
        psi_star = np.asarray(psi_star)[::stride, ::stride]
        top_panels.append(phi_star)   # (H, W)
        bot_panels.append(psi_star)   # (H, W)
        titles_top.append(r'$\phi^\star$ (explicit)')
        titles_bot.append(r'$\psi^\star$ (explicit)')

    # Downsample all time-series panels if requested, and slice to `iters`
    def _ds(p):
        return p[:iters, ::stride, ::stride] if p.ndim == 3 else p
    top_panels = [ _ds(p) for p in top_panels ]
    bot_panels = [ _ds(p) for p in bot_panels ]

    ncols = len(top_panels)
    fig, axs = plt.subplots(2, ncols, figsize=(3.4*ncols, 6), dpi=dpi, constrained_layout=True)

    # Precompute color normalization
    def _norm_for(p):
        if p.ndim == 3:
            vmin = float(jnp.min(p))
            vmax = float(jnp.max(p))
        else:
            vmin = float(np.min(p)); vmax = float(np.max(p))
        if vmin == vmax:
            eps = 1e-12; vmin -= eps; vmax += eps
        return Normalize(vmin=vmin, vmax=vmax)

    if use_global_norm:
        norms_top = [ _norm_for(p) for p in top_panels ]
        norms_bot = [ _norm_for(p) for p in bot_panels ]
    else:
        norms_top = [ Normalize() for _ in top_panels ]
        norms_bot = [ Normalize() for _ in bot_panels ]

    # Create artists
    ims = []
    for j, (p, t, nrm) in enumerate(zip(top_panels, titles_top, norms_top)):
        a0 = p[0] if (hasattr(p, "ndim") and getattr(p, "ndim", 2) == 3) else p
        im = axs[0, j].imshow(np.asarray(a0), origin="lower", aspect="auto", cmap=cmap,
                              norm=nrm, interpolation="nearest", animated=True)
        axs[0, j].set_title(t, fontsize=10)
        axs[0, j].tick_params(labelsize=8)
        ims.append(im)

    for j, (p, t, nrm) in enumerate(zip(bot_panels, titles_bot, norms_bot)):
        a0 = p[0] if (hasattr(p, "ndim") and getattr(p, "ndim", 2) == 3) else p
        im = axs[1, j].imshow(np.asarray(a0), origin="lower", aspect="auto", cmap=cmap,
                              norm=nrm, interpolation="nearest", animated=True)
        axs[1, j].set_title(t, fontsize=10)
        axs[1, j].tick_params(labelsize=8)
        ims.append(im)

    # Add one colorbar per row (attached to the first image in the row)
    fig.colorbar(ims[0],  ax=axs[0, :].tolist(), shrink=0.85)
    fig.colorbar(ims[ncols], ax=axs[1, :].tolist(), shrink=0.85)

    # Update logic
    def _set_frame(im, panel, k, norm):
        if hasattr(panel, "ndim") and getattr(panel, "ndim", 2) == 3:
            frame = np.asarray(panel[k])
            im.set_data(frame)
            if not use_global_norm:
                vmin = float(np.min(frame)); vmax = float(np.max(frame))
                if vmin == vmax:
                    eps = 1e-12; vmin -= eps; vmax += eps
                im.set_clim(vmin=vmin, vmax=vmax)

    def update(k):
        # top row
        for j in range(ncols):
            _set_frame(ims[j],         top_panels[j], k, norms_top[j])
        # bottom row
        base = ncols
        for j in range(ncols):
            _set_frame(ims[base + j],  bot_panels[j], k, norms_bot[j])

        fig.suptitle(f"Back-and-Forth internal variables — iteration {k}/{iters-1}", fontsize=12)
        return ims

    anim = FuncAnimation(
        fig,
        update,
        frames=iters,
        interval=interval,
        blit=True,
        cache_frame_data=False,
        save_count=iters
    )

    if save_path:
        ext = save_path.split(".")[-1].lower()
        fps = max(1, int(1000 / interval))
        if ext == "gif":
            # Pillow writer (no external binaries)
            anim.save(save_path, fps=fps, writer="pillow")
        elif ext == "mp4":
            # Requires ffmpeg; if unavailable, fall back to pillow gif
            try:
                anim.save(save_path, fps=fps, dpi=dpi)
            except Exception:
                alt = save_path.rsplit(".", 1)[0] + ".gif"
                print(f"[warn] ffmpeg not found, saving GIF instead: {alt}")
                anim.save(alt, fps=fps, writer="pillow")
        else:
            # default to GIF
            alt = save_path + ".gif"
            anim.save(alt, fps=fps, writer="pillow")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return anim

def get_caffarelli_counterexample_grid(
    n_points: int,
    *,
    center=(0.5, 0.5),
    r_src=0.22,             # radius of source BALL
    r_in=0.12,              # inner radius of target ANNULUS
    r_out=0.35,             # outer radius of target ANNULUS
    soften=0.0              # optional soft boundary (0 = hard indicator)
):
    """
    Build a Caffarelli-style counterexample:
      μ = uniform on a disk  B(center, r_src)
      ν = uniform on an annulus A(center; r_in, r_out)

    Both are discretized on a cell-centered [0,1]^2 grid of shape (n_points, n_points).

    Args:
      soften: if >0, uses a smooth "soft indicator" (sigmoid shell) to avoid
              razor-sharp edges; useful for numerics on coarse grids.
    Returns:
      axes (list[jnp.ndarray of shape (n_points,)]), mu (H,W), nu (H,W)
    """
    # cell-centered axes in [0,1]
    L0, L1 = 0.0, 1.0
    length = L1 - L0
    h = length / n_points
    ax = jnp.linspace(L0 + 0.5*h, L1 - 0.5*h, n_points)
    axes = [ax, ax]

    X, Y = jnp.meshgrid(ax, ax, indexing="ij")
    cx, cy = center
    R2 = (X - cx)**2 + (Y - cy)**2
    R = jnp.sqrt(R2)

    if soften <= 0.0:
        # hard indicators
        # mu_mask = (R <= r_src).astype(jnp.float32)
        # nu_mask = ((R >= r_in) & (R <= r_out)).astype(jnp.float32)
        mu_mask = (R <= r_src).astype(jnp.float64)
        nu_mask = ((R >= r_in) & (R <= r_out)).astype(jnp.float64)
    else:
        # soft transitions around each boundary (sigmoid with width ~ soften)
        # helpful if you see ringing/instability right on the boundary
        def smooth_step(x, edge, width):
            # ~sigmoid((edge - x)/width) for "x <= edge"
            return 1.0 / (1.0 + jnp.exp((x - edge) / width))

        inner = 1.0 - smooth_step(R, r_in, soften)   # ~ 1 for R >= r_in
        outer =      smooth_step(R, r_out, soften)   # ~ 1 for R <= r_out
        annulus = inner * outer

        ball = smooth_step(R, r_src, soften)         # ~ 1 for R <= r_src
        # mu_mask = ball.astype(jnp.float32)
        # nu_mask = annulus.astype(jnp.float32)
        mu_mask = ball.astype(jnp.float64)
        nu_mask = annulus.astype(jnp.float64)

    # normalize to make probability densities on the grid
    mu = mu_mask / mu_mask.sum()
    nu = nu_mask / nu_mask.sum()
    return axes, mu, nu


def get_translated_ball_grid(
    n_points: int,
    *,
    center_src=(0.5, 0.5),   # source ball center
    radius=0.10,             # ball radius
    delta=(0.15, 0.00),      # translation vector for target ball
    soften=0.0,              # soft boundary width (0 = hard indicator)
    # Non-convex support (annulus) options:
    restrict_to_annulus=True,
    ann_center=(0.5, 0.5),
    r_in=0.12,
    r_out=0.35,
):
    """
    Build μ, ν on a cell-centered [0,1]^2 grid (n_points x n_points):
      μ = uniform on a disk B(center_src, radius)
      ν = uniform on a disk B(center_src + delta, radius)

    If restrict_to_annulus is True, both are intersected with the same annulus
    A(ann_center; r_in, r_out). Choose centers/radius so each ball lies fully
    inside the annulus to avoid clipping (defaults are safe).
    """
    # cell-centered axes in [0,1]
    L0, L1 = 0.0, 1.0
    h = (L1 - L0) / n_points
    ax = jnp.linspace(L0 + 0.5 * h, L1 - 0.5 * h, n_points)
    axes = [ax, ax]

    X, Y = jnp.meshgrid(ax, ax, indexing="ij")

    # distances to ball centers
    cx, cy = center_src
    tx, ty = (cx + delta[0], cy + delta[1])

    R_src = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    R_tgt = jnp.sqrt((X - tx) ** 2 + (Y - ty) ** 2)

    # (optional) annulus mask
    if restrict_to_annulus:
        acx, acy = ann_center
        R_ann = jnp.sqrt((X - acx) ** 2 + (Y - acy) ** 2)
        ann_mask = (R_ann >= r_in) & (R_ann <= r_out)
    else:
        ann_mask = jnp.ones_like(X, dtype=bool)

    # hard/soft ball indicators
    if soften <= 0.0:
        mu_mask = (R_src <= radius)
        nu_mask = (R_tgt <= radius)
    else:
        # smooth step ~1 for r <= radius, width ~ soften
        def smooth_in_ball(R, r, w):
            return 1.0 / (1.0 + jnp.exp((R - r) / w))
        mu_mask = smooth_in_ball(R_src, radius, soften) > 0.5
        nu_mask = smooth_in_ball(R_tgt, radius, soften) > 0.5

    mu_mask = mu_mask & ann_mask
    nu_mask = nu_mask & ann_mask

    # convert to float densities and normalize
    mu = mu_mask.astype(jnp.float64)
    nu = nu_mask.astype(jnp.float64)

    # guard against empty supports
    mu = jnp.where(mu.sum() > 0, mu / mu.sum(), mu)
    nu = jnp.where(nu.sum() > 0, nu / nu.sum(), nu)

    return axes, mu, nu


def get_two_balls_grid(
    n_points: int,
    *,
    center_mu=(0.5, 0.5),
    center_nu=(0.7, 0.5),
    r: float = 0.12,
    box=(0.0, 1.0),
    dtype=jnp.float64,
):
    """
    Build μ and ν as uniform disks (same radius r) on a cell-centered [box[0], box[1]]^2 grid.

    Args:
      n_points: number of cells per axis (produces n_points x n_points grid)
      center_mu: (cx, cy) center of μ's ball
      center_nu: (cx, cy) center of ν's ball
      r: ball radius for both μ and ν
      box: (L0, L1) domain limits for both axes
      dtype: array dtype (default float32)

    Returns:
      axes: [ax, ax] (each shape (n_points,))
      mu: jnp.ndarray (n_points, n_points), μ density normalized to sum=1
      nu: jnp.ndarray (n_points, n_points), ν density normalized to sum=1
    """
    L0, L1 = box
    h = (L1 - L0) / n_points
    ax = jnp.linspace(L0 + 0.5 * h, L1 - 0.5 * h, n_points, dtype=dtype)
    axes = [ax, ax]

    X, Y = jnp.meshgrid(ax, ax, indexing="ij")

    # μ ball
    cx_mu, cy_mu = center_mu
    R_mu = jnp.sqrt((X - cx_mu) ** 2 + (Y - cy_mu) ** 2)
    mu_mask = (R_mu <= r).astype(dtype)

    # ν ball
    cx_nu, cy_nu = center_nu
    R_nu = jnp.sqrt((X - cx_nu) ** 2 + (Y - cy_nu) ** 2)
    nu_mask = (R_nu <= r).astype(dtype)

    # Normalize (guard against empty masks)
    mu_sum = mu_mask.sum()
    nu_sum = nu_mask.sum()
    if (mu_sum == 0) or (nu_sum == 0):
        raise ValueError(
            "Empty support: check centers/radius so each ball intersects the grid."
        )

    mu = mu_mask / mu_sum
    nu = nu_mask / nu_sum
    return axes, mu, nu


def get_line_and_square_grid(
    n_points: int,
    *,
    mu_is: str = "line",                 # "line" or "square"
    # Line segment params (in box coordinates):
    line_endpoints=((0.20, 0.50), (0.80, 0.50)),
    line_width: float | None = None,     # if None, set to ~1.5*h
    # Square params:
    square_center=(0.50, 0.50),
    square_side: float = 0.30,           # side length
    # Domain:
    box=(0.0, 1.0),
    dtype=jnp.float64,
    soften: float = 0.0,                 # boundary softness; 0 = hard mask
):
    """
    Build μ and ν supports:
      - 'line': thin rectangle around a segment between two endpoints, width = line_width
      - 'square': axis-aligned filled square of given center & side

    If mu_is = "line": μ = line, ν = square. Otherwise swapped.

    soften > 0 adds a smooth transition around boundaries (sigmoid width ~ soften*h).
    """
    assert mu_is in ("line", "square")
    L0, L1 = box
    h = (L1 - L0) / n_points
    if line_width is None:
        line_width = 1.5 * h  # ~1–2 pixels thick by default

    # cell-centered grid
    ax = jnp.linspace(L0 + 0.5 * h, L1 - 0.5 * h, n_points, dtype=dtype)
    axes = [ax, ax]
    X, Y = jnp.meshgrid(ax, ax, indexing="ij")

    # ---- line mask (distance-to-segment <= line_width/2) ----
    (x1, y1), (x2, y2) = line_endpoints
    A = jnp.array([x1, y1], dtype=dtype)
    B = jnp.array([x2, y2], dtype=dtype)
    P = jnp.stack([X, Y], axis=-1)                 # (..., 2)
    v = B - A
    vv = jnp.dot(v, v) + 1e-30                     # avoid 0-length
    t = jnp.clip(jnp.sum((P - A) * v, axis=-1) / vv, 0.0, 1.0)
    Q = A + t[..., None] * v                       # closest point on segment
    dist = jnp.linalg.norm(P - Q, axis=-1)
    if soften <= 0:
        line_mask = (dist <= (line_width * 0.5)).astype(dtype)
    else:
        # smooth step ~1 inside, width ~ soften*h
        w = soften * h
        line_mask = 1.0 / (1.0 + jnp.exp((dist - (line_width * 0.5)) / w))
        line_mask = (line_mask > 0.5).astype(dtype)  # keep boolean-like behavior

    # ---- square mask (axis-aligned) ----
    cx, cy = square_center
    half = 0.5 * square_side
    if soften <= 0:
        square_mask = (
            (jnp.abs(X - cx) <= half) &
            (jnp.abs(Y - cy) <= half)
        ).astype(dtype)
    else:
        w = soften * h
        def sstep(a):  # smooth indicator for |a| <= half
            return 1.0 / (1.0 + jnp.exp((jnp.abs(a) - half) / w))
        square_mask = (sstep(X - cx) * sstep(Y - cy))
        square_mask = (square_mask > 0.5).astype(dtype)

    # assign μ, ν
    mu_mask = line_mask if mu_is == "line" else square_mask
    nu_mask = square_mask if mu_is == "line" else line_mask

    # normalize and guard empties
    mu_sum = mu_mask.sum()
    nu_sum = nu_mask.sum()
    if (mu_sum == 0) or (nu_sum == 0):
        raise ValueError("Empty support: adjust endpoints/width or square position/size.")

    mu = mu_mask / mu_sum
    nu = nu_mask / nu_sum
    return axes, mu, nu



if __name__ == "__main__":
    L0, L1 = 0.0, 1.0

    def cell_centered_axes(n_points: int):
        length = L1 - L0
        h = length / (n_points)
        ax = jnp.linspace(L0+0.5*h, L1-0.5*h, n_points)
        return ax

    def get_gaussian_grid(n_points: int, means: list, covs: list):
        dim = len(means)
        mean = jnp.asarray(means)
        cov = jnp.asarray(covs)
        invCov = jnp.linalg.inv(cov)
        ax = cell_centered_axes(n_points)
        axes = [ax for _ in range(dim)]
        grid_axes = [ax.ravel() for ax in axes]
        meshgrids = jnp.meshgrid(*grid_axes, indexing='ij')
        X = jnp.stack(meshgrids, axis=-1)
        def pdf(X):
            diff = X - mean
            quad = jnp.einsum('...i,ij,...j', diff, invCov, diff)
            dens = jnp.exp(-0.5 * quad)
            dens = dens / dens.sum()
            return dens
        return axes, pdf(X)

    # n_points = 64
    # m1 = [0.25, 0.25]
    # m2 = [0.75, 0.75]
    # m1 = [0.4, 0.4]
    # m2 = [0.6, 0.6]
    # m1 = [0.2, 0.2]
    # m2 = [0.8, 0.8]
    # S1 = [[0.01, 0], [0, 0.01]]
    # S2 = [[0.01, 0], [0, 0.01]]
    # S1 = [[0.01, 0], [0, 0.02]]
    # S2 = [[0.02, 0], [0, 0.01]]
    # S1 = [[0.03, 0.001], [0, 0.02]]
    # S2 = [[0.04, 0], [0.01, 0.02]]

    # m1 = [0.5, 0.5]
    # m2 = [0.5, 0.5]
    # S1 = [[0.001, 0], [0, 0.02]]
    # S2 = [[0.02, 0], [0, 0.001]]

    # axes, mu = get_gaussian_grid(
    #     n_points=n_points,
    #     means=m1,
    #     covs=S1,
    # )
    # _, nu = get_gaussian_grid(
    #     n_points=n_points,
    #     means=m2,
    #     covs=S2,
    # )
    # mu /= mu.sum()
    # nu /= nu.sum()
    # print(f"mu sum = {mu.sum()}, nu sum = {nu.sum()}")

    # OR USE CAFFARELLI COUNTEREXAMPLE
    # n_points = 128

    # axes, mu, nu = get_caffarelli_counterexample_grid(
    #     n_points=n_points,
    #     center=(0.5, 0.5),
    #     r_src=0.22,
    #     r_in=0.12,
    #     r_out=0.35,
    #     soften=0.0,   # try 0.5*h or 1.0*h if you want soft edges
    # )

    # JUST BALL TRANSLATION
    # n_points = 128
    # axes, mu, nu = get_translated_ball_grid(
    #     n_points=n_points,
    #     center_src=(0.5 + 0.22, 0.5),  # sit on the ring
    #     radius=0.08,                   # small enough to fit the annulus
    #     delta=(0.00, 0.20),            # translate along the ring
    #     restrict_to_annulus=True,      # keep the same non-convex support
    #     r_in=0.12, r_out=0.35,
    # )
    n_points = 128
    axes, mu, nu = get_two_balls_grid(
        n_points,
        center_mu=(0.40, 0.50),
        center_nu=(0.65, 0.55),
        r=0.10
    )

    # A LINE TO THE SQUARE
    # n_points = 64
    # axes, mu, nu = get_line_and_square_grid(
    #     n_points,
    #     mu_is="line",
    #     line_endpoints=((0.2, 0.2), (0.2, 0.8)),
    #     line_width=None,              # auto ~1.5*h
    #     square_center=(0.6, 0.5),
    #     square_side=0.15,
    # )

    X0, X1 = jnp.meshgrid(*axes, indexing='ij')
    points = jnp.stack([X0.ravel(), X1.ravel()], axis=1)


    start = time.perf_counter()
    maxiters = 300
    stepsize = 8.0 / jnp.maximum(mu.max(), nu.max())
    error_metric = 'tv_psi'
    tolerance = 1e-5
    progressbar = False
    # 1) Run your solver as you already do:
    results = backnforth_sqeuclidean_nd(
        mu=mu, nu=nu, coordinates=axes,
        # stepsize=30,
        stepsize=stepsize,
        maxiterations=maxiters,
        tolerance=tolerance,
        progressbar=progressbar,
        stepsize_lower_bound=0.01,
        error_metric=error_metric,
        # error_metric="transportation_cost_relative"
    )
    end = time.perf_counter()
    # print(f"Computed in {time.perf_counter() - start}s")

    # (Optional) If you computed explicit Gaussian φ*, ψ* for comparison:
    # phi_star = phi_explicit   # shape (H, W)
    # psi_star = psi_explicit   # shape (H, W)
    # compare = (phi_star, psi_star)
    compare = None

    # 2) Make & save the animation
    _ = make_bfm_animation(
        results,
        interval=400,            # faster playback
        # save_path="bfm.mp4",
        save_path="bfm-ball-to-ball.mp4",
        # save_path="bfm-line-to-square.gif",
        show=False,
        stride=1,               # set to 2 or 3 to downsample frames for speed
        use_global_norm=True,   # stable colorbars across time
        dpi=120,
        # cmap='Greys',
        # compare=compare         # or None
    )

    
    start2 = time.perf_counter()
    results2 = bfm_uot(
        mu=mu,
        nu=nu,
        coordinates=axes,
        stepsize=stepsize,
        maxiterations=maxiters,
        tolerance=tolerance,
        progressbar=progressbar,
        error_metric=error_metric,
    )
    end2 = time.perf_counter()

    print(f"Iterations on solver 1: {results[0]}")
    print(f"Iterations on solver 2: {results2[0]}")

    print(f"Time of the first solver: {end - start}s")
    print(f"Time of the second solver {end2- start2}s")

    monge_map1 = monge_map_from_psi_nd(results[2])
    monge_map2 = monge_map_from_psi_nd(-results2[2])

    # previous computation of the monge map 
    # shape = mu.shape
    # d = mu.ndim
    # n_vec = jnp.array(shape, dtype=jnp.float32)
    # grids = jnp.meshgrid(*axes, indexing="ij")     # list of d arrays, each (*shape)
    # X = jnp.stack(grids, axis=-1)                     # (*shape, d)
    # grad_psi = _central_gradient_nd(-results2[2])              # (d, *shape)
    # grad_psi = grad_psi * n_vec.reshape((-1,) + (1,) * d)
    # if grad_psi.shape[0] == len(axes):
    #     grad_psi = jnp.moveaxis(grad_psi, 0, -1)      # -> (*shape, d)
    # monge_map2 = jnp.clip(
    #     X - grad_psi,
    #     # X + grad_psi,           # potential is negative
    #     0, 1)

    # phi1, phi2 = results[1], results2[1]

    # print(f"Iterations {results[0]} - {results2[0]}")

    # results_close = jnp.allclose(results[1], results2[1], rtol=1e-8)
    # print(f"Potential phi close {results_close}")
    # print(f"{jnp.max(jnp.abs(phi1 - phi2))=}")

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(results[4])
    # axs[1].imshow(results2[4])
    # fig.savefig('bfm_method_pushforwards.png', dpi=300)
    # plt.close(fig)

    # k1 = int(results[0])  - 1
    # k2 = int(results2[0]) - 1

    # psi_ct_1 = np.asarray(results[10][k1])   # (H,W) — solver 1, ψ after c-transform
    # psi_ct_2 = np.asarray(results2[10][k2])  # (H,W) — solver 2, ψ after c-transform

    psi_ct_1 = np.asarray(results[2])
    psi_ct_2 = np.asarray(results2[2])

    X0, X1   = _grid_from_axes(axes)
    h0, h1   = _spacings_from_axes(axes)

    # Tψ = X - ∇ψ  (μ -> ν)
    grad_psi_1 = _central_gradient_2d(psi_ct_1, h0, h1)
    grad_psi_2 = _central_gradient_2d(psi_ct_2, h0, h1)

    Tpsi0_1 = X0 - grad_psi_1[..., 0];  Tpsi1_1 = X1 - grad_psi_1[..., 1]
    Tpsi0_2 = X0 - grad_psi_2[..., 0];  Tpsi1_2 = X1 - grad_psi_2[..., 1]

    # displacements for quiver
    U0_1 = Tpsi0_1 - X0;  U1_1 = Tpsi1_1 - X1
    U0_2 = Tpsi0_2 - X0;  U1_2 = Tpsi1_2 - X1

    # downsample quiver for readability
    qstep = 6  # try 6–8 for sparser arrows on large grids
    X0q, X1q, U0q_1, U1q_1 = _downsample_quiver(X0, X1, U0_1, U1_1, step=qstep)
    _,   _,   U0q_2, U1q_2 = _downsample_quiver(X0, X1, U0_2, U1_2, step=qstep)

    extent = [[ax.min(), ax.max()] for ax in axes]
    extent = [item for sublist in extent for item in sublist]

    # --- build the figure: keep your pushforward images and overlay quiver -------

    fig, axs = plt.subplots(3, 2, figsize=(10, 10), dpi=150, constrained_layout=True)

    # Left: pushforward of solver 1 (you had results[4])
    CONTOUR_LINES_ALPHA = 0.6
    vmax = np.max([
        mu.max(),
        nu.max(),
        np.asarray(results2[4]).max(),
    ])
    # im0 = axs[0, 0].imshow(np.asarray(results[4]), origin="lower",
    #                        extent=extent,
    #                        cmap="viridis")
    # axs[0, 0].set_title("Solver 1: pushforward with $T_\\psi$ overlay")
    # axs[0, 0].set_aspect("equal", adjustable="box")

    # axs[0,0].contour(X0, X1, mu, alpha=CONTOUR_LINES_ALPHA,
    #                  linewidths=1.2)
    # axs[0,0].contour(X0, X1, nu, alpha=CONTOUR_LINES_ALPHA,
    #                  linewidths=1.2)
    # im1 = axs[0,0].imshow(nu, extent=extent, origin='lower')

    print(f"pushforard sum {np.asarray(results2[4]).sum()}")
    # Right: pushforward of solver 2 (you had results2[4])
    im2 = axs[0, 1].imshow(np.asarray(results2[4]), origin="lower",
                           extent=extent,
                           cmap="viridis", vmax=vmax)
    # axs[0, 1].set_title("Solver 2: pushforward with $T_\\psi$ overlay")
    axs[0, 1].set_title("pushforward with $T_\\psi$ overlay")
    axs[0, 1].set_aspect("equal", adjustable="box")

    # axs[0,1].contour(X0, X1, mu, alpha=CONTOUR_LINES_ALPHA,
    #                  linewidths=1.2)
    # axs[0,1].contour(X0, X1, nu, alpha=CONTOUR_LINES_ALPHA,
    #                  linewidths=1.2)

    im1 = axs[0,0].imshow(mu, extent=extent, origin='lower', vmax=vmax)
    axs[0,0].set_title('Measure $\\mu$')
    im3 = axs[1,0].imshow(nu, extent=extent, origin='lower', vmax=vmax)
    axs[1,0].set_title('Measure $\\nu$')

    fig.colorbar(im3, ax=axs)

    # Overlay quiver on both
    # Tweak scale/width if arrows look too long/short
    # axs[1, 0].quiver(X0q, X1q, U0q_1, U1q_1, angles="xy", scale_units="xy", scale=1.0, width=0.003)
    axs[1, 1].quiver(X0q, X1q, U0q_2, U1q_2, angles="xy", scale_units="xy", scale=1.0, width=0.003)

    # or plot the monge map
    # print(f"{monge_map1.shape=}")
    # print(f"{monge_map2.shape=}")
    # im3 = axs[1,0].imshow(monge_map2[0], origin='lower')
    # fig.colorbar(im3, ax=axs[1,0])
    # im4 = axs[1,1].imshow(monge_map2[1], origin='lower')
    # fig.colorbar(im4, ax=axs[1,1])

    # (optional) shared colorbar for the backgrounds
    # fig.colorbar(im0, ax=axs, shrink=0.85)

    error_range1 = slice(1, results[0])
    error_range2 = slice(1, results2[0])
    # axs[2,0].plot(list(range(results[0])[error_range1]), results[5][error_range1], c='green', label='Sol.1')
    axs[2,0].plot(list(range(results2[0])[error_range2]), results2[5][error_range2], c='red', label='Sol.2')
    # axs[2,0].set_title(f"Error (error diff between methods {results[5][results[0]] - results2[5][results2[0]]})")
    axs[2,0].set_title('TV distance error')
    axs[2,0].set_yscale('log')
    axs[2,0].grid()
    axs[2,0].legend()
    # axs[2,1].plot(results[17][:results[0]], c='green', label='Sol.1')
    axs[2,1].plot(results2[7][:results[0]], c='red', label='Sol.2')
    axs[2,1].set_title("Stepsize")
    axs[2,1].set_yscale('log')
    axs[2,1].grid()
    axs[2,1].legend()

    fig.savefig("bfm_method_pushforwards_with_quiver.png", dpi=300)
    plt.close(fig)