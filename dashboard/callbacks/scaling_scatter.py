import dash
from dash import Input, Output
import plotly.express as px

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP


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
        dff.groupby(["solver", "size"], as_index=False)
           .agg(mean_runtime=("runtime", "mean"))
    )

    # line plot with markers
    fig = px.line(
        agg.sort_values("size"),
        x="size",
        y="mean_runtime",
        color="solver",
        markers=True,
        title="Mean Runtime vs Problem Size",
        labels={
            "size": "Problem Size (# points)",
            "mean_runtime": "Mean Runtime (s)"
        },
        color_discrete_map=SOLVERS_COLOR_MAP,
    )

    # log–log axes
    fig.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=sorted(agg["size"].unique()),
        ticktext=[str(int(v)) for v in sorted(agg["size"].unique())],
        ticks="outside",
        tickangle=45
    )
    fig.update_yaxes(type="log", ticks="outside")

    # tighten layout
    fig.update_layout(
        template="plotly_white",
        legend_title="Solver",
        margin=dict(l=60, r=20, t=50, b=60),
        height=450,
    )

    return fig
