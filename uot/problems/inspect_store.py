import argparse
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from uot.problems.store import ProblemStore
from uot.problems.iterator import ProblemIterator

from uot.utils.logging import setup_logger

logger = setup_logger('inspect_store')
logger.propagate = False

# NOTE: ensure you have kaleido installed: pip install -U kaleido


def plot_1d(mu_pts, mu_w, nu_pts, nu_w):
    Scatter = go.Scatter
    fig = go.Figure()
    fig.add_trace(Scatter(x=mu_pts.flatten(), y=mu_w,
                          mode='markers+lines', name='μ'))
    fig.add_trace(Scatter(x=nu_pts.flatten(), y=nu_w,
                          mode='markers+lines', name='ν'))
    return fig


def plot_2d(mu_pts, mu_w, nu_pts, nu_w):
    Scatter = go.Scatter
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('μ points', 'ν points'),
                        horizontal_spacing=0.1)
    fig.add_trace(Scatter(x=mu_pts[:, 0], y=mu_pts[:, 1], mode='markers',
                          marker=dict(size=4, opacity=0.6), name='μ'), row=1, col=1)
    fig.add_trace(Scatter(x=nu_pts[:, 0], y=nu_pts[:, 1], mode='markers',
                          marker=dict(size=4, opacity=0.6), name='ν'), row=1, col=2)
    return fig


def plot_nd(mu_pts, mu_w, nu_pts, nu_w):
    # project onto 2D with PCA
    from sklearn.decomposition import PCA
    all_pts = np.vstack([mu_pts, nu_pts])
    proj = PCA(n_components=2).fit_transform(all_pts)
    mu_proj = proj[:len(mu_pts)]
    nu_proj = proj[len(mu_pts):]

    Scatter = go.Scatter
    fig = go.Figure()
    fig.add_trace(Scatter(x=mu_proj[:, 0], y=mu_proj[:, 1], mode='markers',
                          marker=dict(size=3, opacity=0.6), name='μ (PCA)'))
    fig.add_trace(Scatter(x=nu_proj[:, 0], y=nu_proj[:, 1], mode='markers',
                          marker=dict(size=3, opacity=0.6), name='ν (PCA)'))
    fig.update_layout(title_text='PCA Projection of points')
    return fig


def plot_and_save(mu_pts, mu_w, nu_pts, nu_w, out_prefix):
    mu_pts = np.atleast_2d(mu_pts)
    nu_pts = np.atleast_2d(nu_pts)
    dim = mu_pts.shape[1]

    # select plotting function
    plot_funcs = {1: plot_1d, 2: plot_2d}
    plot_func = plot_funcs.get(dim, plot_nd)

    # choose webgl trace for large data
    total_pts = mu_pts.shape[0] + nu_pts.shape[0]
    if total_pts > 20000:
        logger.info(
            f"[{out_prefix}] The number of points is too large. Using WebGL.")
        for func in [plot_1d, plot_2d, plot_nd]:
            func.__globals__['Scatter'] = go.Scattergl
    else:
        for func in [plot_1d, plot_2d, plot_nd]:
            func.__globals__['Scatter'] = go.Scatter

    fig = plot_func(mu_pts, mu_w, nu_pts, nu_w)

    fig.update_layout(
        width=1000,
        height=600,
        title=out_prefix,
        legend=dict(x=0.01, y=0.99)
    )

    # requires kaleido
    img_path = out_prefix + ".png"
    fig.write_image(img_path)

    # and save interactive html
    html_path = out_prefix + ".html"
    fig.write_html(html_path)

    logger.info(f"Saved {img_path} and {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect & save dataset visualizations.")
    parser.add_argument(
        "--store", type=str, required=True,
        help="Path to the storage file to inspect."
    )
    parser.add_argument(
        "--outdir", type=str, default="plots",
        help="Directory to write plot images and HTML."
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    store = ProblemStore(args.store)
    iterator = ProblemIterator(store)

    for idx, problem in enumerate(iterator):
        mu, nu = problem.get_marginals()
        mu_pts, mu_w = mu.to_discrete()
        nu_pts, nu_w = nu.to_discrete()

        prefix = os.path.join(args.outdir, f"problem_{idx:04d}")
        plot_and_save(mu_pts, mu_w, nu_pts, nu_w, prefix)
