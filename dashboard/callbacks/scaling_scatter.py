import dash
from dash import Input, Output
import plotly.graph_objects as go

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP

# Predefine a list of marker symbols for regs (add more as needed)
MARKER_SYMBOLS = [
    "circle", "square", "diamond", "cross", "x", "triangle-up",
    "triangle-down", "pentagon", "star", "hourglass", "bowtie",
    "triangle-left", "triangle-right", "hexagram", "star-diamond"
]


@dash.callback(
    Output("desc-scaling-scatter", "figure"),
    *FILTER_INPUTS,
    Input("desc-scaling-scatter-only-converged", "value"),
)
def update_scaling_scatter(
        solvers, regs, dims, datasets, size, only_converged
):
    dff = get_filtered(solvers, regs, dims, datasets, size, only_converged)
    agg = (
        dff.groupby(["solver", "size", "reg"], as_index=False)
           .agg(median_runtime=("runtime", "median"))
    )

    unique_regs = sorted(agg["reg"].unique())
    reg_to_marker = {
        reg: MARKER_SYMBOLS[i % len(MARKER_SYMBOLS)]
        for i, reg in enumerate(unique_regs)
    }

    fig = go.Figure()
    for solver in sorted(agg["solver"].unique()):
        for reg in unique_regs:
            subset = agg[(agg["solver"] == solver) & (agg["reg"] == reg)].sort_values("size")
            if not subset.empty:
                fig.add_trace(
                    go.Scatter(
                        x=subset["size"],
                        y=subset["median_runtime"],
                        mode="lines+markers",
                        name=f"{solver}, reg={reg}",
                        marker=dict(
                            symbol=reg_to_marker[reg],
                            size=9, opacity=0.85,
                            color=SOLVERS_COLOR_MAP.get(solver, "#888")),
                        line=dict(color=SOLVERS_COLOR_MAP.get(solver, "#888"),
                                  width=2),
                        hovertemplate=(
                                    f"Solver: {solver}<br>"
                                    f"Reg: {reg}<br>"
                                    "Size: %{x}<br>"
                                    "Median runtime: %{y:.3f}s<extra></extra>"),
                    )
                )

    fig.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=sorted(agg["size"].unique()),
        ticktext=[str(int(v)) for v in sorted(agg["size"].unique())],
        ticks="outside",
        tickangle=45
    )
    fig.update_yaxes(type="log", ticks="outside")

    fig.update_layout(
        template="plotly_white",
        legend_title="Solver, reg",
        margin=dict(l=60, r=20, t=50, b=60),
        height=450,
    )

    return fig
