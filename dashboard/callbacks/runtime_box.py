import dash
from dash import Output
import plotly.express as px
from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP



@dash.callback(
    Output("desc-runtime-box", "figure"),
    *FILTER_INPUTS,
)
def update_runtime_box(solvers, regs, dims, datasets, size):
    dff = get_filtered(solvers, regs, dims, datasets, size)
    fig = px.box(
        dff, x="solver", y="runtime", color="solver",
        title="Runtime by Solver",
        color_discrete_map=SOLVERS_COLOR_MAP,
    )
    return fig
