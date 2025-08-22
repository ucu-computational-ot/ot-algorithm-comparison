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
        df.groupby(["solver", "size"], dropna=False)
          .agg(cost_rerr=("cost_rerr", "median"))
          .reset_index()
    )

    fig = go.Figure()

    solver_list = list(dict.fromkeys(summary["solver"].tolist()))
    for sol in solver_list:
        d = summary[summary["solver"] == sol]
        color = SOLVERS_COLOR_MAP[sol]
        d = d.sort_values("size")
        fig.add_trace(
            go.Scatter(
                x=d["size"],
                y=d["cost_rerr"],
                mode="lines+markers",
                name=sol,
                # line=dict(width=2),
                # marker=dict(size=6),

                fillcolor=color,
                line=dict(color=color, width=2),
                marker=dict(size=7, opacity=0.9, color=color),

                hovertemplate=(
                    f"solver: {sol}<br>"
                    "size: %{x}<br>"
                    "cost error: %{y:.3e}<extra></extra>"
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
        xaxis=dict(title="Problem size (#points)", type="log"),
        yaxis=dict(title="Cost relative error", type="log"),
        margin=dict(l=70, r=20, t=60, b=60),
        height=500,
        template="plotly_white",
        legend=dict(title="Solver"),
        font=dict(size=14),
    )

    return fig
