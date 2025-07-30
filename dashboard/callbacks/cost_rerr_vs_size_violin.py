import dash
from dash import Output
import plotly.express as px
from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP


@dash.callback(
    Output('desc-cost-err-problem-size-violonplot', "figure"),
    *FILTER_INPUTS,
)
def update_cost_rerr_violinplot(solvers, regs, dims, datasets, size):
    dff = get_filtered(solvers, regs, dims, datasets, size, converged=True)
    fig = px.violin(
        dff,
        x="cost_rerr",
        y="dataset",
        color="solver",
        orientation="h",
        box=True,
        points="all",
        hover_data=["iterations", "reg", "solver"],
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
        title="Solver Cost RERR Across All Datasets",
        xaxis_title="Cost Relative Error",
        yaxis_title="Dataset",
        legend_title="Solver",
        template="plotly_white",
    )
    return fig
