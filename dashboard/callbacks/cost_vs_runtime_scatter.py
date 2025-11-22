import dash
from dash import Output
import plotly.express as px
from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP


@dash.callback(
    Output("desc-cost-error-vs-runtime", "figure"),
    *FILTER_INPUTS,
)
def update_cost_error_vs_runtime_plot(solvers, regs, dims, datasets, size):
    dff = get_filtered(solvers, regs, dims, datasets, size)

    fig = px.scatter(
        dff,
        # x="runtime",
        # y="cost_rerr",

        y="runtime",
        x="cost_rerr",

        color="solver",
        color_discrete_map=SOLVERS_COLOR_MAP,
        symbol="dim",
        title="Cost Relative Error vs Runtime",
        marginal_x="histogram",
        marginal_y="violin",
        log_x=True,
        # log_y=True,
        template="plotly_white"
    )
    # fig.add_hline(y=1e-6, line_dash="dash", line_color="gray")
    fig.update_xaxes(
        tickformat=".1e",             # e.g. “2.0e-05” instead of “20 µ”
        exponentformat="e",
        showexponent="all",
        # type="log",
    )
    fig.update_yaxes(
        tickformat=".1e",             # e.g. “2.0e-05” instead of “20 µ”
        exponentformat="e",
        showexponent="all",
        # type="log",
    )
    return fig
