import dash
from dash import Output
import plotly.express as px
from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MAX_COLS = 3         # wrap solver facets after this many columns
RUNTIME_LOG = True   # runtimes are usually heavy-tailed → log scale helps


@dash.callback(
    Output('desc-runtime-violinplot', "figure"),
    *FILTER_INPUTS,
)
def update_runtime_violonplot(solvers, regs, dims, datasets, size):
    df = get_filtered(solvers, regs, dims, datasets, size, converged=True)

    # short-circuit
    if (df.empty or not
            {"solver", "distribution", "size", "runtime"} <= set(df.columns)):
        return go.Figure(layout=dict(
            title="Solver Timings Across All Datasets",
            annotations=[dict(text="No data", x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False)]
        ))

    # ---- tidy ordering ----
    def _to_num(v):
        try:
            return float(v)
        except Exception:
            return np.inf
    size_order = sorted(df["size"].unique(),
                        key=lambda x: (_to_num(x), str(x)))
    dist_order = sorted(df["distribution"].unique(), key=str)
    solver_list = list(dict.fromkeys(df["solver"].tolist()))
    n_sol = len(solver_list)
    n_cols = min(MAX_COLS, max(1, n_sol))
    n_rows = int(np.ceil(n_sol / n_cols))

    # ---- figure scaffold ----
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        # shared_yaxes=True,
        shared_xaxes=False,
        horizontal_spacing=0.04,
        vertical_spacing=0.10 if n_rows > 1 else 0.08,
        subplot_titles=solver_list,
    )

    # color palette for distributions (consistent across panels)
    # (choose any qualitative palette; Set2 is calm and paper-friendly)
    dist_palette = px.colors.qualitative.Set1
    color_map = {
        d: dist_palette[i % len(dist_palette)]
        for i, d in enumerate(dist_order)
    }

    for i, solver in enumerate(solver_list):
        r, c = i // n_cols + 1, i % n_cols + 1
        d = df[df["solver"] == solver].copy()

        # ensure consistent category order panel-to-panel
        d["size"] = pd.Categorical(d["size"], categories=size_order, ordered=True)
        d["distribution"] = pd.Categorical(d["distribution"], categories=dist_order, ordered=True)

        # produce one violin per (distribution,size)
        # offsetgroup = size so violins for the same size stand together
        # legendgroup = distribution so color/legend is shared across panels
        for di, dist in enumerate(dist_order):
            dd = d[d["distribution"] == dist]
            if dd.empty:
                continue

        for di, dist in enumerate(dist_order):
            dd = d[d["distribution"] == dist]
            if dd.empty:
                continue
            fig.add_trace(
                go.Violin(
                    x=dd["size"].astype(str),
                    y=dd["runtime"],
                    name=str(dist),
                    legendgroup=str(dist),
                    scalegroup=str(dist),
                    # offsetgroup="size",           # groups by size bins

                    offsetgroup=str(dist),        # unique per distribution
                    alignmentgroup="size",        # align widths within each size bin

                    fillcolor=color_map[dist],  # ✅ force legend swatch to match fill

                    marker=dict(color=color_map[dist], opacity=0.6),
                    line=dict(width=1, color="rgba(0,0,0,0.25)"),
                    meanline=dict(visible=False),

                    side="both",       # force half-violin side
                    spanmode="hard",       # consistent bandwidth

                    box=dict(visible=True, width=0.4),   # inner box for quartiles
                    points="outliers",
                    # points="all",
                    pointpos=0.0,                 # center the jitter over the violin
                    jitter=0.15,
                    opacity=0.9,
                    hovertemplate=(
                        f"solver: {solver}<br>"
                        "distribution: %{fullData.name}<br>"
                        "size: %{x}<br>"
                        "runtime: %{y:.4g}s<extra></extra>"
                    ),
                    showlegend=(i == 0),         # show legend once, top-left panel
                ),
                row=r, col=c
            )

        # axis cosmetics per panel
        fig.update_xaxes(
            title_text="Size" if r == n_rows else None,
            categoryorder="array",
            categoryarray=[str(s) for s in size_order],
            tickangle=0,
            row=r, col=c
        )
        if c == 1:
            fig.update_yaxes(title_text="Runtime (s)", row=r, col=c)

    # ---- shared layout ----
    # dynamic height: depends on rows
    base_per_row = 300
    height = base_per_row * n_rows

    fig.update_layout(
        height=height,
        margin=dict(l=90, r=20, t=60, b=60),
        # title="Solver runtimes by size and distribution (grouped violins per size)",
        title="",
        template="plotly_white",
        legend_title_text="Distribution",
        violinmode="group",        # group violins within each size bin
        boxmode="overlay",
        hovermode="closest",
    )

    # log-scale runtime helps heavy tails; clamp zeros safely
    if RUNTIME_LOG:
        for ax_i in range(1, n_rows * n_cols + 1):
            fig.layout[f"yaxis{'' if ax_i==1 else ax_i}"].update(type="log", rangemode="normal")
        fig.add_annotation(
            text="log scale", x=1.0, y=1.02, xref="paper", yref="paper",
            xanchor="right", showarrow=False, font=dict(size=12, color="gray")
        )

    # tighter spacing if many solvers
    # if n_rows > 1 or n_cols > 2:
    #     fig.update_layout(horizontal_spacing=0.035)

    # make it “trackable”: align y-axes & add light reference grid
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", gridwidth=0.5)

    return fig

    # fig = px.violin(
    #     df,
    #     x="runtime",
    #     y="dataset",
    #     color="solver",
    #     orientation="h",
    #     box=True,
    #     points="all",
    #     hover_data=["iterations", "reg"],
    #     color_discrete_map=SOLVERS_COLOR_MAP,
    # )
    # fig.update_traces(
    #     width=0.8,
    # )
    # N = len(df['dataset'].unique())
    # per_row = 50
    # base = 200
    # fig.update_layout(
    #     height=base + per_row * N,
    #     margin=dict(l=200, r=20, t=50, b=50),
    #     title="Solver Timings Across All Datasets",
    #     xaxis_title="Time (s)",
    #     yaxis_title="Dataset",
    #     legend_title="Solver",
    #     template="plotly_white",
    # )
    # return fig
