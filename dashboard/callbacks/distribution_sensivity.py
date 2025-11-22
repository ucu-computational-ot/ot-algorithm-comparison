import dash
from dash import Output
import numpy as np
import pandas as pd
from plotly import express as px
import plotly.graph_objects as go

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered

# Optional: turn on log scale for cost error
LOG_Y = True

try:
    from utils.data import SOLVERS_COLOR_MAP
except Exception:
    SOLVERS_COLOR_MAP = {}

PREFERRED_FAMILIES = ["Gaussian", "Student", "Cauchy", "GH"]


@dash.callback(
    Output("desc-dist-sensitivity", "figure"),
    *FILTER_INPUTS,
)
def update_distribution_sensitivity(solvers, regs, dims, datasets, size):
    # Use converged runs for cost error comparison
    df = get_filtered(solvers, regs, dims, datasets, size, converged=True)

    required = {"distribution", "solver", "cost_rerr"}
    if df.empty or not required.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(
            title="Distribution Family Sensitivity",
            annotations=[dict(text="No data for current filters",
                              x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    dff = df.copy()
    dff["cost_rerr"] = pd.to_numeric(dff["cost_rerr"], errors="coerce")
    dff = dff.dropna(subset=["cost_rerr", "distribution", "solver"])

    # Preferred family order if present; otherwise alphabetical
    families_present = dff["distribution"].dropna().unique().tolist()
    ordered = [f for f in PREFERRED_FAMILIES if f in families_present]
    for f in sorted(set(families_present) - set(ordered)):
        ordered.append(f)
    dff["distribution"] = pd.Categorical(dff["distribution"], categories=ordered, ordered=True)

    # Build figure with one box trace per (solver) grouped by dataset category
    fig = go.Figure()
    solver_order = list(dict.fromkeys(dff["solver"].tolist()))

    # color per solver (fallback palette if not in map)
    fallback = px.colors.qualitative.Set2
    solvers_order = list(dict.fromkeys(dff["solver"].tolist()))
    solver_colors = {
        s: SOLVERS_COLOR_MAP.get(s, fallback[i % len(fallback)])
        for i, s in enumerate(solvers_order)
    }

    # Small positive clip for log scale (if needed)
    tiny = 1e-16
    if LOG_Y:
        dff = dff[dff["cost_rerr"] > 0]
        dff["cost_rerr"] = dff["cost_rerr"].clip(lower=tiny)

    for sol in solver_order:
        dd = dff[dff["solver"] == sol]
        if dd.empty:
            continue

        color = solver_colors[sol]
        # trace_kwargs = {}
        # if color is not None:
            # trace_kwargs["marker_color"] = color
            # trace_kwargs["line"] = dict(color="rgba(0,0,0,0.25)", width=1)

        fig.add_trace(
            go.Box(
                x=dd["distribution"],
                y=dd["cost_rerr"],
                name=sol,
                boxmean=True,               # show mean line inside box (median always shown)
                jitter=0.3,                 # jitter individual points

                fillcolor=color,
                line=dict(color="rgba(0,0,0,0.35)", width=1),
                marker=dict(size=4, opacity=0.55, color=color),

                pointpos=0.0,
                whiskerwidth=0.8,
                boxpoints="outliers",       # show outliers to see heavy tails
                notched=False,
                # marker=dict(size=4, opacity=0.5),
                hovertemplate=(
                    "solver: " + sol + "<br>"
                    "distribution: %{x}<br>"
                    "cost error: %{y:.3e}<extra></extra>"
                ),
                offsetgroup=sol,            # group boxes by solver color/legend
                alignmentgroup="distribution",   # line up boxes across families
                # alignmentgroup="dataset",   # line up boxes across families
                # **trace_kwargs
            )
        )

    # X ticks: show all families in chosen order
    fig.update_xaxes(
        title="Dataset family",
        categoryorder="array",
        categoryarray=ordered,
        tickangle=0
    )

    fig.update_yaxes(
        title="Cost relative error vs LP baseline",
        type="log" if LOG_Y else "linear"
    )

    fig.update_layout(
        # title="Distribution Family Sensitivity (boxplots by family, colored by solver)",
        margin=dict(l=80, r=20, t=60, b=60),
        template="plotly_white",
        legend=dict(title="Solver"),
        font=dict(size=14),
        boxmode="group"  # ensure grouped by family
    )

    # Optional interpretive note (reviewer-friendly): heavy-tailed families → wider boxes/outliers
    # fig.add_annotation(
    #     text="Note: heavier tails (e.g., Student/Cauchy/GH) often yield larger error/instability for Sinkhorn-like methods.",
    #     x=0.0, y=1.12, xref="paper", yref="paper", xanchor="left",
    #     showarrow=False, font=dict(size=12, color="gray")
    # )

    return fig
