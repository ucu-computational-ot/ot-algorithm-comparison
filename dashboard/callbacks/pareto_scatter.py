import dash
from dash import Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered

# optional: use your solver color map if available
try:
    from utils.data import SOLVERS_COLOR_MAP
except Exception:
    SOLVERS_COLOR_MAP = {}


@dash.callback(
    Output("desc-pareto-scatter", "figure"),
    *FILTER_INPUTS,
)
def update_pareto_scatter(solvers, regs, dims, datasets, size):
    # pull filtered data; use converged runs for meaningful error/time
    df = get_filtered(solvers, regs, dims, datasets, size, converged=True)

    # be robust to schema differences: runtime may be 'runtime' or 'time'
    runtime_col = "runtime" if "runtime" in df.columns else ("time" if "time" in df.columns else None)
    required = {"solver", "cost_rerr"}
    if df.empty or runtime_col is None or not required.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(
            title="Pareto scatter: runtime vs cost error",
            annotations=[dict(text="No data for current filters",
                              x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    dff = df.copy()
    dff[runtime_col] = pd.to_numeric(dff[runtime_col], errors="coerce")
    dff["cost_rerr"] = pd.to_numeric(dff["cost_rerr"], errors="coerce")


    # keep valid, strictly positive for log axes
    dff = dff.dropna(subset=[runtime_col, "cost_rerr"])
    dff = dff[(dff[runtime_col] > 0) & (dff["cost_rerr"] > 0)]

    if dff.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Pareto scatter: runtime vs cost error",
            annotations=[dict(text="No positive runtime and error after filtering",
                              x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    # ---------- compute overall Pareto frontier (minimize both) ----------
    # sort by runtime; keep points whose error is a new running minimum
    pts = dff[[runtime_col, "cost_rerr"]].to_numpy()
    order = np.argsort(pts[:, 0])
    pts_sorted = pts[order]

    pareto = []
    best_err = np.inf
    for x, y in pts_sorted:
        if y < best_err:
            pareto.append((x, y))
            best_err = y
    pareto = np.array(pareto)

    # ---------- build scatter figure ----------
    fig = go.Figure()

    # color points by solver
    solver_order = list(dict.fromkeys(dff["solver"].tolist()))
    regs = sorted(list(df['reg'].unique()))
    MARKER_MAP = {
        reg: symbol for reg, symbol in zip(
            sorted(df['reg'].unique()),
            ['circle', 'square', 'diamond', 'cross', 'star', 'x', 'hexagram',
             'triangle-up', 'circle-cross', 'square-cross']
        )}
    for s in solver_order:
        for reg in regs:
            dd = dff[(dff["solver"] == s) & (dff['reg'] == reg)]
            symbol = MARKER_MAP[reg]
            color = SOLVERS_COLOR_MAP.get(s, None)
            fig.add_trace(
                go.Scatter(
                    x=dd[runtime_col],
                    y=dd["cost_rerr"],
                    mode="markers",
                    name=f"{s}, reg={reg}",
                    marker=dict(size=6, opacity=0.6, color=color,
                                symbol=symbol,
                                line=dict(width=1, color='Black')),
                    hovertemplate=(
                        f"solver: {s}<br>"
                        f"reg: {reg}<br>"
                        "runtime: %{x} s<br>"
                        "cost error: %{y:.3e}<br>"
                        + ("size: %{customdata[0]}<br>" if "size" in dd.columns else "")
                        # + ("reg: %{customdata[1]}<br>" if "reg" in dd.columns else "")
                        + ("iters: %{customdata[2]}<br>" if "iterations" in dd.columns else "")
                        + "<extra></extra>"
                    ),
                    # pass a few useful columns in customdata if they exist
                    customdata=np.stack([
                        dd["size"] if "size" in dd.columns else pd.Series([None]*len(dd)),
                        dd["reg"] if "reg" in dd.columns else pd.Series([None]*len(dd)),
                        dd["iterations"] if "iterations" in dd.columns else pd.Series([None]*len(dd)),
                    ], axis=-1) if any(c in dd.columns for c in ["size", "reg", "iterations"]) else None,
                )
            )

    # overlay Pareto frontier as a polyline (sorted by runtime)
    if pareto.size >= 2:
        fig.add_trace(
            go.Scatter(
                x=pareto[:, 0],
                y=pareto[:, 1],
                mode="lines+markers",
                name="Pareto frontier",
                line=dict(width=3, color="black"),
                marker=dict(size=8, color="black"),
                hovertemplate="Pareto point<br>runtime: %{x} s<br>cost error: %{y:.3e}<extra></extra>",
            )
        )
    else:
        # single best point fallback
        bp = pareto[0] if pareto.size == 2 else None
        if bp is not None:
            fig.add_trace(
                go.Scatter(
                    x=[bp[0]], y=[bp[1]],
                    mode="markers",
                    name="Pareto best",
                    marker=dict(size=10, color="black", symbol="star"),
                    hovertemplate="Pareto best<br>runtime: %{x} s<br>cost error: %{y:.3e}<extra></extra>",
                )
            )

    fig.update_layout(
        # title="Pareto scatter: runtime vs cost error (lower-left is better)",
        xaxis=dict(title="Runtime (s)", type="log"),
        yaxis=dict(title="Cost relative error", type="log"),
        # yaxis=dict(title="Cost relative error vs LP baseline", type="log"),
        template="plotly_white",
        margin=dict(l=80, r=20, t=60, b=60),
        legend=dict(title="Solver"),
        height=560,
        hovermode="closest",
    )

    # optional guiding note
    # fig.add_annotation(
    #     text="Pareto frontier connects non-dominated runs (min runtime & error).",
    #     x=0.0, y=1.12, xref="paper", yref="paper", xanchor="left",
    #     showarrow=False, font=dict(size=12, color="gray")
    # )

    return fig
