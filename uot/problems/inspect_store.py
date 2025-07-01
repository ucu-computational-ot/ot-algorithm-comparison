import argparse
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from uot.problems.store import ProblemStore
from uot.problems.iterator import ProblemIterator

import logging
from uot.utils.logging import setup_logger

logger = setup_logger('inspect_store')
logger.propagate = False
logger.setLevel(
    logging.DEBUG if os.environ.get('DEBUG', False) else logging.INFO)

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

    logger.debug(f"Saved {img_path} and {html_path}")


def plot_store(store_path: str, outdir: str):
    store = ProblemStore(store_path)
    iterator = ProblemIterator(store)
    logger.debug(f"Beginning iteration over {store}")

    for idx, problem in enumerate(iterator):
        logger.debug(f"Processing problem {problem}")
        mu, nu = problem.get_marginals()
        mu_pts, mu_w = mu.to_discrete()
        nu_pts, nu_w = nu.to_discrete()

        prefix = os.path.join(outdir, f"problem_{idx:04d}")
        logger.debug(f"Output files would have prefix {prefix}")
        plot_and_save(mu_pts, mu_w, nu_pts, nu_w, prefix)
    logger.info(f"Processed {len(iterator)} problems in {store}")


def plot_dataset(dataset_path: str, outdir: str):
    path = Path(dataset_path)
    outpath = Path(outdir)
    logger.debug(f"Dataset path is {path}")
    logger.debug(f"Output path is {outpath}")
    stores = [entry for entry in path.iterdir() if entry.is_dir()]
    for store in stores:
        store_output = outpath / store.name
        os.makedirs(store_output, exist_ok=True)
        logger.debug(f"Created store output directory in {store_output}")
        plot_store(store, store_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect & save dataset visualizations.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--store", type=str, default=None,
        help="Path to the storage file to inspect."
    )
    input_group.add_argument(
        "--dataset", type=str, default=None,
        help="Path to the dataset location to inspect."
    )
    parser.add_argument(
        "--outdir", type=str, default="plots",
        help="Directory to write plot images and HTML."
    )
    args = parser.parse_args()

    logger.debug(f"Creating output directory {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)

    if args.store is not None:
        logger.debug(f"Flag set to visualize store. Will read {args.store} and\
        output to {args.outdir}")
        plot_store(args.store, args.outdir)
    elif args.dataset is not None:
        logger.debug(f"Flag set to visualize dataset. Will read {args.dataset}\
        and output to {args.outdir}")
        plot_dataset(args.dataset, args.outdir)
    else:
        raise ValueError("No input data option (either store or \
        dataset) was specified.")
