import dash
from dash import Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered
from utils.data import SOLVERS_COLOR_MAP

from utils.perf_profile import (
    INSTANCE_KEYS,
    TAU_MAX_DEFAULT,
    N_TAU,
    R_M,
    _runtime_column,
    _compute_ratios_per_run,
    _profile_curves_runs,
)

# ---------------------------
# Shared helpers
# ---------------------------

# INSTANCE_KEYS = ["distribution", "size", "reg", "dim"]
# INSTANCE_KEYS = ["distribution", "size", "dim"]
# TAU_MAX_DEFAULT = 100.0
# N_TAU = 200
MAX_COLS = 3
# R_M = 1e6  # large ratio for missing/failed runs

# def _runtime_column(df: pd.DataFrame) -> str:
#     if "runtime" in df.columns: return "runtime"
#     if "time" in df.columns:    return "time"
#     raise KeyError("Neither 'runtime' nor 'time' present in dataframe.")
#
#
# def _compute_ratios(df: pd.DataFrame, runtime_col: str, r_M: float = R_M) -> pd.DataFrame:
#     """
#     Dolan–Moré ratios with *median* aggregation over repeats.
#     Missing/invalid runs for a (solver, instance) are encoded as ratio = r_M.
#
#     Steps
#     -----
#     1) Median runtime per (solver, INSTANCE_KEYS) from repeats.
#        (Only positive, finite times are valid here.)
#     2) For each instance, find the best (min) valid median across solvers.
#     3) Build the full solver×instance grid:
#          - ratio = median / best_median  (where median is valid)
#          - ratio = r_M                   (where median missing/invalid)
#     4) Drop instances where best_median is missing (no solver had a valid run).
#
#     Returns
#     -------
#     DataFrame with columns: ['solver', 'ratio'] + INSTANCE_KEYS
#     """
#     cols_needed = ["solver", runtime_col] + INSTANCE_KEYS
#     d = df[cols_needed].copy()
#
#     # numeric cleaning
#     d[runtime_col] = pd.to_numeric(d[runtime_col], errors="coerce")
#
#     # ---- 1) median per (solver, instance) from valid repeats ----
#     valid = d[np.isfinite(d[runtime_col]) & (d[runtime_col] > 0)].copy()
#     if valid.empty:
#         # nothing valid anywhere → nothing to normalize
#         return pd.DataFrame(columns=["solver", "ratio"] + INSTANCE_KEYS)
#
#     per_solver_inst = (
#         valid.groupby(["solver"] + INSTANCE_KEYS, dropna=False, sort=False)[runtime_col]
#              .median()
#              .rename("median_runtime")
#              .reset_index()
#     )
#
#     # ---- 2) best median per instance (across solvers) ----
#     best_per_inst = (
#         per_solver_inst.groupby(INSTANCE_KEYS, dropna=False, sort=False)["median_runtime"]
#                        .min()
#                        .rename("best_median_runtime")
#                        .reset_index()
#     )
#
#     # Keep only instances with at least one valid solver (i.e., best is finite & > 0)
#     best_per_inst = best_per_inst[
#         np.isfinite(best_per_inst["best_median_runtime"]) &
#         (best_per_inst["best_median_runtime"] > 0)
#     ]
#     if best_per_inst.empty:
#         return pd.DataFrame(columns=["solver", "ratio"] + INSTANCE_KEYS)
#
#     # ---- 3) build full solver × instance grid, then left-join medians ----
#     # unique solvers that *appear anywhere* (so we include “missing solver on instance” cases)
#     all_solvers = pd.Index(df["solver"].dropna().astype(str).unique())
#     # all instances that are *normalizable*
#     all_instances = best_per_inst[INSTANCE_KEYS].drop_duplicates()
#
#     # cartesian product (grid)
#     grid = (
#         all_instances.assign(_tmp=1)
#         .merge(pd.DataFrame({"solver": all_solvers, "_tmp": 1}), on="_tmp")
#         .drop(columns="_tmp")
#     )
#
#     # attach medians (may be NaN if solver missing on that instance)
#     grid = grid.merge(
#         per_solver_inst,
#         on=["solver"] + INSTANCE_KEYS,
#         how="left",
#         validate="many_to_one",
#     ).merge(
#         best_per_inst,
#         on=INSTANCE_KEYS,
#         how="left",
#         validate="many_to_one",
#     )
#
#     # compute ratios
#     # valid where median is finite & > 0
#     valid_mask = np.isfinite(grid["median_runtime"]) & (grid["median_runtime"] > 0)
#     grid["ratio"] = r_M  # default to failure cap
#     grid.loc[valid_mask, "ratio"] = (
#         grid.loc[valid_mask, "median_runtime"] / grid.loc[valid_mask, "best_median_runtime"]
#     )
#
#     # tidy return
#     out = grid[["solver", "ratio"] + INSTANCE_KEYS].copy()
#     # guard: ratio must be positive and finite; if not (shouldn't happen), set to r_M
#     bad = ~np.isfinite(out["ratio"]) | (out["ratio"] <= 0)
#     if bad.any():
#         out.loc[bad, "ratio"] = r_M
#
#     return out.reset_index(drop=True)


# def _choose_tau_grid(ratios: pd.DataFrame, tau_max: float | None) -> np.ndarray:
#     if ratios.empty:
#         return np.array([1.0, 1.1])
#     if tau_max is None:
#         # robust upper bound: cap by default, but don’t squash genuine spread
#         finite_max = np.nanmax(ratios["ratio"].values)
#         # if extreme, use a quantile to avoid a single outlier blowing out the axis
#         q95 = np.nanquantile(ratios["ratio"].values, 0.95)
#         upper = float(np.clip(max(min(finite_max, q95), 2.0), 2.0, TAU_MAX_DEFAULT))
#         tau_max = upper
#     tau_max = max(1.01, float(tau_max))
#     return np.geomspace(1.0, tau_max, N_TAU)


# def _profile_curves(ratios: pd.DataFrame, group_keys=None, tau_max=None):
#     """
#     Build Φ_s(τ) for each panel/group.
#     Returns: dict {group_key_tuple: (taus, {solver: Φ(τ)})}, and a stable solver order.
#     """
#     if ratios.empty:
#         return {}, []
#
#     taus = _choose_tau_grid(ratios, tau_max)
#
#     # global stable solver order (first-appearance order)
#     global_solver_order = list(dict.fromkeys(ratios["solver"].tolist()))
#     results = {}
#
#     if not group_keys:
#         groups = [((), ratios)]
#     else:
#         groups = list(ratios.groupby(group_keys, dropna=False))
#
#     for gkey, gdf in groups:
#         # instances available in THIS panel
#         inst = gdf[INSTANCE_KEYS].drop_duplicates()
#         N = len(inst)
#         if N == 0:
#             # no instances → empty profiles
#             results[gkey] = (taus, {})
#             continue
#
#         # solvers actually present in THIS panel
#         solvers_here = list(dict.fromkeys(gdf["solver"].tolist()))
#         profiles = {}
#
#         # compute Φ_s(τ) = (1/N) * |{i : r_{s,i} ≤ τ}|
#         # ratios are already one row per (solver, instance)
#         for s in solvers_here:
#             r = gdf.loc[gdf["solver"] == s, "ratio"].to_numpy()
#             if r.size == 0:
#                 continue
#             counts = (r[None, :] <= taus[:, None]).sum(axis=1)
#             profiles[s] = counts / float(N)
#
#         results[gkey] = (taus, profiles)
#
#     return results, global_solver_order


def _add_profile_traces(fig, taus, profiles, solver_order,
                        panel_name=None, show_legend=True,
                        **kwargs):
    panel_prefix = f"{panel_name}<br>" if panel_name else ""
    for s in solver_order:
        y = profiles.get(s)
        if y is None:
            continue
        color = SOLVERS_COLOR_MAP.get(s, None)   # fallback to None if missing
        fig.add_trace(
            go.Scatter(
                x=taus,
                y=y,
                mode="lines",
                line_shape="hv",
                name=s,
                line=dict(width=2, color=color),
                hovertemplate=(
                    panel_prefix
                    + "solver: %{fullData.name}<br>"
                    + "τ: %{x:.3g}<br>"
                    # + "P($\\tau$): %{y:.3f}<extra></extra>"
                    + "P(τ): %{y:.3f}<extra></extra>"
                ),
                showlegend=show_legend,
            ),
            **kwargs
        )
        # fig.update_xaxes(type="log",
                 # tickvals=[1,1.5,2,3,5,10,20,50,100],
                 # title="τ (relative runtime factor, log scale)",
                         # )
        # for v in [2,5]:
        #     fig.add_vline(x=v, line_width=1, opacity=0.2)


# ---------------------------
# 1) Performance profile (overall): grouped by solver
#     Output id: "desc-perfprof-solver"
# ---------------------------

@dash.callback(
    Output("desc-perfprof-solver", "figure"),
    *FILTER_INPUTS,
)
def perf_profile_by_solver(solvers, regs, dims, datasets, size):
    df = get_filtered(solvers, regs, dims, datasets, size, converged=True)
    try:
        rt_col = _runtime_column(df)
    except KeyError:
        return go.Figure(layout=dict(
            title="Performance Profile (by solver)",
            annotations=[dict(text="No runtime column", x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False)]
        ))

    ratios = _compute_ratios_per_run(df, runtime_col=_runtime_column(df))
    results, solver_order = _profile_curves_runs(ratios, group_keys=None)
    # ratios = _compute_ratios(df[["solver", rt_col] + INSTANCE_KEYS], rt_col)
    # results, solver_order = _profile_curves(ratios, group_keys=None)

    fig = go.Figure()
    if results:
        taus, profiles = list(results.values())[0]
        _add_profile_traces(fig, taus, profiles, solver_order, show_legend=True)
        # --- Make higher-performing lines thicker (simple & robust) ---
        scores = [float(np.nanmean(np.asarray(tr.y, dtype=float))) for tr in fig.data]
        smin, smax = min(scores), max(scores)
        for tr, sc in zip(fig.data, scores):
            width = 1 if smax == smin else 1.5 + 1.1 * (sc - smin) / (smax - smin)
            # width = 2.0 if smax == smin else 1.5 + 4.5 * (sc - smin) / (smax - smin)
            tr.update(line=dict(width=width))

    fig.update_layout(
        # title="Performance Profile — overall (Φ(τ) by solver)",
        xaxis=dict(title=r"$\tau$ (relative runtime)", type="log"),
        yaxis=dict(title=r"$P(\tau)$  (fraction of instances)", range=[-0.03, 1.03]),
        template="plotly_white",
        margin=dict(l=80, r=20, t=60, b=60),
        height=520,
        legend_title_text="Solver",
    )
    return fig

# ---------------------------
# 2) Performance profile: grouped by solver & PROBLEM SIZE
#     Facet by size, curves per solver
#     Output id: "desc-perfprof-solver-size"
# ---------------------------

@dash.callback(
    Output("desc-perfprof-solver-size", "figure"),
    *FILTER_INPUTS,
)
def perf_profile_by_solver_size(solvers, regs, dims, datasets, size):
    df = get_filtered(solvers, regs, dims, datasets, size, converged=True)
    try:
        rt_col = _runtime_column(df)
    except KeyError:
        return go.Figure(layout=dict(
            title="Performance Profile (by solver & size)",
            annotations=[dict(text="No runtime column", x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False)]
        ))

    # ratios = _compute_ratios(df[["solver", rt_col] + INSTANCE_KEYS], rt_col)
    ratios = _compute_ratios_per_run(df, runtime_col=_runtime_column(df))

    # Determine size order (numeric if possible)
    def _to_num(v):
        try: return float(v)
        except Exception: return np.inf
    size_order = sorted(ratios["size"].dropna().unique(), key=lambda x: (_to_num(x), str(x)))

    # results, solver_order = _profile_curves(ratios, group_keys=["size"])
    results, solver_order = _profile_curves_runs(ratios, group_keys=["size"])

    # build facets
    panels = [(k if isinstance(k, tuple) else (k,), v) for k, v in results.items()]
    # sort panels by size order
    order_map = {s:i for i,s in enumerate(size_order)}
    panels.sort(key=lambda kv: order_map.get(kv[0][0], 1e9))

    n_panels = len(panels)
    if n_panels == 0:
        return go.Figure(layout=dict(
            title="Performance Profile (by solver & size)",
            annotations=[dict(text="No data after filtering", x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False)]
        ))
    n_cols = min(MAX_COLS, max(1, n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[rf'$n = {str(k[0])}$' for k,_ in panels],
        horizontal_spacing=0.06,
        vertical_spacing=0.12 if n_rows > 1 else 0.08,
        shared_yaxes=True
    )

    for i, ((k,), (taus, profiles)) in enumerate(panels):
        r, c = i // n_cols + 1, i % n_cols + 1
        _add_profile_traces(fig, taus, profiles, solver_order,
                            panel_name=rf'$n={k}$', show_legend=(i==0),
                            row=r, col=c)

        # --- Make higher-performing lines thicker (simple & robust) ---
        scores = [float(np.nanmean(np.asarray(tr.y, dtype=float))) for tr in fig.data]
        smin, smax = min(scores), max(scores)
        for tr, sc in zip(fig.data, scores):
            width = 1 if smax == smin else 1.5 + 1.0 * (sc - smin) / (smax - smin)
            # width = 2.0 if smax == smin else 1.5 + 4.5 * (sc - smin) / (smax - smin)
            tr.update(line=dict(width=width))

        fig.update_xaxes(type="log", title_text=r"$\tau$", row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text=r"$P(\tau)$", range=[-0.03,1.03], row=r, col=c)

    height = max(360, n_rows * 300)
    fig.update_layout(
        # title="Performance Profile — by problem size (curves per solver)",
        template="plotly_white",
        margin=dict(l=80, r=20, t=60, b=60),
        height=height,
        legend_title_text="Solver",
    )
    return fig

# ---------------------------
# 3) Performance profile: grouped by solver & DISTRIBUTION
#     Facet by distribution, curves per solver
#     Output id: "desc-perfprof-solver-dist"
# ---------------------------

@dash.callback(
    Output("desc-perfprof-solver-dist", "figure"),
    *FILTER_INPUTS,
)
def perf_profile_by_solver_distribution(solvers, regs, dims, datasets, size):
    df = get_filtered(solvers, regs, dims, datasets, size, converged=True)
    try:
        rt_col = _runtime_column(df)
    except KeyError:
        return go.Figure(layout=dict(
            title="Performance Profile (by solver & distribution)",
            annotations=[dict(text="No runtime column", x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False)]
        ))

    # ratios = _compute_ratios(df[["solver", rt_col] + INSTANCE_KEYS], rt_col)
    ratios = _compute_ratios_per_run(df, runtime_col=_runtime_column(df))

    # distribution order (alphabetical)
    dist_order = sorted(ratios["distribution"].dropna().unique(), key=str)

    # results, solver_order = _profile_curves(ratios, group_keys=["distribution"])
    results, solver_order = _profile_curves_runs(ratios, group_keys=["distribution"])

    panels = [(k if isinstance(k, tuple) else (k,), v) for k, v in results.items()]
    # sort by preferred order
    order_map = {d:i for i,d in enumerate(dist_order)}
    panels.sort(key=lambda kv: order_map.get(kv[0][0], 1e9))

    n_panels = len(panels)
    if n_panels == 0:
        return go.Figure(layout=dict(
            title="Performance Profile (by solver & distribution)",
            annotations=[dict(text="No data after filtering", x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False)]
        ))
    n_cols = min(MAX_COLS, max(1, n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{str(k[0])}" for k,_ in panels],
        horizontal_spacing=0.06,
        vertical_spacing=0.12 if n_rows > 1 else 0.08,
        shared_yaxes=True
    )

    for i, ((k,), (taus, profiles)) in enumerate(panels):
        r, c = i // n_cols + 1, i % n_cols + 1
        _add_profile_traces(fig, taus, profiles, solver_order,
                            panel_name=f"{k}", show_legend=(i==0),
                            row=r, col=c)
        fig.update_xaxes(type="log", title_text=r"$\tau$", row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text=r"$P(\tau)$", range=[-0.03,1.03], row=r, col=c)

    height = max(360, n_rows * 300)
    fig.update_layout(
        # title="Performance Profile — by distribution (curves per solver)",
        template="plotly_white",
        margin=dict(l=80, r=20, t=60, b=60),
        height=height,
        legend_title_text="Solver",
    )
    return fig
