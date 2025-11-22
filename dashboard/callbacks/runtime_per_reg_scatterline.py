import dash
from dash import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP


@dash.callback(
    Output("desc-reg-runtime", "figure"),            # set this id on your dcc.Graph
    *FILTER_INPUTS,                                  # (solvers, regs, dims, datasets, size)
    Input("desc-reg-runtime-only-converged", "value")
)
def update_runtime_vs_reg_median_iqr(solvers, regs, dims, datasets, size, only_converged):
    # 1) Filter rows based on the current controls
    dff = get_filtered(solvers, regs, dims, datasets, size, only_converged).copy()
    if dff.empty:
        return go.Figure(layout=dict(
            template="plotly_white",
            title="Runtime vs ε (no data after filters)",
            height=480
        ))

    # 2) Ensure numeric types
    dff["runtime"] = pd.to_numeric(dff["runtime"], errors="coerce")
    dff["reg"]     = pd.to_numeric(dff["reg"], errors="coerce")
    dff = dff.dropna(subset=["runtime", "reg"])
    if dff.empty:
        return go.Figure(layout=dict(
            template="plotly_white",
            title="Runtime vs ε (no numeric runtime/reg after filters)",
            height=480
        ))

    # 3) Aggregate by (solver, reg): median + IQR (25th–75th), and count
    def q25(s): return s.quantile(0.25, interpolation="linear")
    def q75(s): return s.quantile(0.75, interpolation="linear")

    agg = (
        dff.groupby(["solver", "reg"], as_index=False)
           .agg(median_runtime=("runtime", "median"),
                q25_runtime   =("runtime", q25),
                q75_runtime   =("runtime", q75),
                n             =("runtime", "size"))
    )

    # Avoid non-positive on log y
    tiny = max(agg["median_runtime"].min(), 1e-12) * 1e-6
    agg["y_lo"] = agg["q25_runtime"].clip(lower=tiny)
    agg["y_hi"] = agg["q75_runtime"].clip(lower=tiny)

    # 4) Build figure: one ribbon (IQR) + median line per solver
    fig = go.Figure()
    color_map = SOLVERS_COLOR_MAP

    for solver, g in agg.sort_values(["solver", "reg"]).groupby("solver"):
        g = g.sort_values("reg")
        color = color_map.get(solver)

        # Lower bound (no fill)
        fig.add_trace(go.Scatter(
            x=g["reg"], y=g["y_lo"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            marker=dict(color=color),
            name=f"{solver} (Q1)"
        ))
        # Upper bound (fill to previous to create a ribbon)
        fig.add_trace(go.Scatter(
            x=g["reg"], y=g["y_hi"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            opacity=0.10,
            # opacity=0.25,
            marker=dict(color=color),
            showlegend=False,
            hoverinfo="skip",
            name=f"{solver} (Q3)"
        ))
        # Median line with markers
        fig.add_trace(go.Scatter(
            x=g["reg"], y=g["median_runtime"],
            mode="lines+markers",
            name=solver,
            line=dict(width=2, color=color),
            marker=dict(size=6, color=color),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "ε (reg): %{x}<br>"
                "median runtime: %{y:.4g}s<br>"
                "IQR: [%{customdata[1]:.4g}, %{customdata[2]:.4g}]<br>"
                "n: %{customdata[3]}<extra></extra>"
            ),
            customdata=np.stack([g["solver"], g["y_lo"], g["y_hi"], g["n"]], axis=1),
        ))

    # 5) Axes & layout
    x_axis_type = "log" if (agg["reg"] > 0).all() else "linear"
    fig.update_xaxes(
        type=x_axis_type,
        ticks="outside",
        title_text=r"Regularization $\epsilon$ ε (reg)"
    )
    fig.update_yaxes(
        type="log",
        ticks="outside",
        title_text="Runtime (s)"
    )
    fig.update_layout(
        template="plotly_white",
        legend_title="Solver",
        margin=dict(l=60, r=20, t=50, b=60),
        height=480,
        # title="Runtime vs Regularization ε — Median with IQR band"
    )
    return fig
