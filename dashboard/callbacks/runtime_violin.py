import dash
from dash import Output
import plotly.express as px
from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP


@dash.callback(
    Output('desc-runtime-violinplot', "figure"),
    *FILTER_INPUTS,
)
def update_runtime_violonplot(solvers, regs, dims, datasets, size):
    dff = get_filtered(solvers, regs, dims, datasets, size, converged=True)
    fig = px.violin(
        dff,
        x="runtime",
        y="dataset",
        color="solver",
        orientation="h",
        box=True,
        points="all",
        hover_data=["iterations", "reg"],
        color_discrete_map=SOLVERS_COLOR_MAP,
    )
    fig.update_traces(
        width=0.8,
    )
    N = len(dff['dataset'].unique())
    per_row = 50
    base = 200
    fig.update_layout(
        height=base + per_row * N,
        margin=dict(l=200, r=20, t=50, b=50),
        title="Solver Timings Across All Datasets",
        xaxis_title="Time (s)",
        yaxis_title="Dataset",
        legend_title="Solver",
        template="plotly_white",
    )
    return fig
