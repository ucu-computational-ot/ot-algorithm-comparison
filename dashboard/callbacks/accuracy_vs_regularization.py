import dash
from dash import Output
import pandas as pd
import plotly.graph_objects as go

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP


@dash.callback(
    Output("desc-accuracy-vs-reg", "figure"),
    *FILTER_INPUTS,
)
def update_accuracy_vs_reg(solvers, regs, dims, datasets, size):
    df = get_filtered(solvers, regs, dims, datasets, size, converged=True)
    required = {"solver", "reg", "cost_rerr"}
    if df.empty or not required.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(
            title="Accuracy vs. Regularization",
            annotations=[dict(text="No data for current filters",
                              x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    df = df.copy()
    df["reg"] = pd.to_numeric(df["reg"], errors="coerce")
    df["cost_rerr"] = pd.to_numeric(df["cost_rerr"], errors="coerce")

    reg_values = sorted([float(reg) for reg in df['reg'].unique()])

    # aggregate: median error per (solver,size)
    summary = (
        df.groupby(["solver", "reg"], dropna=False)
          .agg(cost_rerr=("cost_rerr", "median"))
          .reset_index()
    )

    fig = go.Figure()

    solver_list = list(dict.fromkeys(summary["solver"].tolist()))
    # regs = sorted(list(df['reg'].unique()))
    MARKER_MAP = {
        reg: symbol for reg, symbol in zip(
            reg_values,
            # sorted(df['reg'].unique()),
            ['circle', 'square', 'diamond', 'cross', 'star', 'x', 'hexagram',
             'triangle-up', 'circle-cross', 'square-cross']
        )}
    print(f"{MARKER_MAP=}")
    for sol in solver_list:
        # for reg in regs:
        d = summary[(summary["solver"] == sol)]
        # d = summary[(summary["solver"] == sol) & (summary['reg'] == reg)]
        color = SOLVERS_COLOR_MAP[sol]
        # symbol = MARKER_MAP[reg]
        # d = d.sort_values("size")
        fig.add_trace(
            go.Scatter(
                x=d["reg"],
                y=d["cost_rerr"],
                mode="lines+markers",
                name=f"{sol}",
                # fillcolor=color,
                # opacity=0.7,
                # line=dict(color=color, width=1),
                marker=dict(size=10, opacity=0.9, color=color,
                            line=dict(width=2, color='Black')),
                hovertemplate=(
                    f"solver: {sol}<br>"
                    # f"reg: {reg}<br>"
                    # "size: %{x}<br>"
                    "cost rel. error: %{y:.3e}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=reg_values,
                tickformat=".1e",
            )
        )

    fig.update_layout(
        # title="Accuracy vs. Size (cost error vs LP baseline)",
        xaxis=dict(title=r"Regularization", type="log"),
        yaxis=dict(
            title="Median cost relative error",
            type="log",
            exponentformat="E",   # scientific notation, e.g. 1e-3
            # showexponent="all",   # always show exponents
            ticks="outside",
            # minor=dict(show=True),  # show minor ticks
        ),
        margin=dict(l=70, r=20, t=60, b=60),
        height=500,
        template="plotly_white",
        legend=dict(title="Solver"),
        font=dict(size=14),
    )

    return fig
