import dash
from dash import Output
import plotly.express as px
from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP



@dash.callback(
    Output("desc-runtime-hist", "figure"),
    *FILTER_INPUTS,
)
def update_runtime_hist(solvers, regs, dims, datasets, size):
    dff = get_filtered(solvers, regs, dims, datasets, size)
    fig = px.histogram(
        dff,
        x="runtime",
        color="solver",
        marginal="box",
        nbins=50,
        opacity=0.7,
        title="Runtime Distribution",
        color_discrete_map=SOLVERS_COLOR_MAP,
    )
    fig.update_layout(barmode="overlay")
    return fig
