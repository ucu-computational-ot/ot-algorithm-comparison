import dash
from dash import Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP


@dash.callback(
    Output("desc-accuracy-vs-size", "figure"),
    *FILTER_INPUTS,
)
def update_accuracy_vs_size(solvers, regs, dims, datasets, size):
    df = get_filtered(solvers, regs, dims, datasets, size, converged=True)
    required = {"solver", "size", "cost_rerr"}
    if df.empty or not required.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(
            title="Accuracy vs. Size",
            annotations=[dict(text="No data for current filters",
                              x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    df = df.copy()
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["cost_rerr"] = pd.to_numeric(df["cost_rerr"], errors="coerce")

    size_values = [str(size) for size in df["size"].unique()]

    # aggregate: median error per (solver,size)
    summary = (
        df.groupby(["solver", "reg", "size"], dropna=False)
          .agg(cost_rerr=("cost_rerr", "median"))
          .reset_index()
    )

    fig = go.Figure()

    solver_list = list(dict.fromkeys(summary["solver"].tolist()))
    regs = sorted(list(df['reg'].unique()))
    MARKER_MAP = {
        reg: symbol for reg, symbol in zip(
            sorted(df['reg'].unique()),
            ['circle', 'square', 'diamond', 'cross', 'star', 'x', 'hexagram',
             'triangle-up', 'circle-cross', 'square-cross']
        )}
    for sol in solver_list:
        for reg in regs:
            d = summary[(summary["solver"] == sol) & (summary['reg'] == reg)]
            color = SOLVERS_COLOR_MAP[sol]
            symbol = MARKER_MAP[reg]
            d = d.sort_values("size")
            fig.add_trace(
                go.Scatter(
                    x=d["size"],
                    y=d["cost_rerr"],
                    mode="lines+markers",
                    name=f"{sol}, reg={reg}",
                    fillcolor=color,
                    opacity=0.7,
                    line=dict(color=color, width=1),
                    marker=dict(size=10, symbol=symbol, opacity=0.9, color=color,
                                line=dict(width=2, color='Black')),
                    hovertemplate=(
                        f"solver: {sol}<br>"
                        f"reg: {reg}<br>"
                        "size: %{x}<br>"
                        "cost rel. error: %{y:.3e}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=size_values,
                )
            )

    fig.update_layout(
        # title="Accuracy vs. Size (cost error vs LP baseline)",
        xaxis=dict(title="Problem size", type="log"),
        yaxis=dict(title="Median cost relative error", type="log"),
        margin=dict(l=70, r=20, t=60, b=60),
        height=500,
        template="plotly_white",
        legend=dict(title="Solver"),
        font=dict(size=14),
    )

    return fig
