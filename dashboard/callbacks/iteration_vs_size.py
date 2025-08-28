import dash
from dash import Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP


@dash.callback(
    Output("desc-iteration-vs-problem-size", "figure"),
    *FILTER_INPUTS,
)
def update_iterations_boxplot(solvers, regs, dims, datasets, size):
    dff = get_filtered(solvers, regs, dims, datasets, size)
    n_solvers = len(solvers)

    # precompute the ordered categories as strings
    ordered_sizes = [str(n) for n in sorted(dff["size"].astype(int).unique())]

    fig = make_subplots(
        rows=1,
        cols=n_solvers,
        subplot_titles=solvers,
        shared_yaxes=True,
        x_title="Problem Size (# points)",
        y_title="Iterations",
    )

    for col_idx, solver in enumerate(solvers, start=1):
        subset = dff[dff["solver"] == solver]
        # convert to string so it's treated categorically
        x_cat = subset["size"].astype(int).astype(str)
        color = SOLVERS_COLOR_MAP.get(solver, "#444")

        fig.add_trace(
            go.Box(
                x=x_cat,
                y=subset["iterations"],
                name=solver,
                boxpoints="all",
                jitter=0.3,
                # marker=dict(size=4, opacity=0.6),
                # line=dict(width=1),
                marker=dict(color=color, size=4, opacity=0.6),
                line=dict(color=color, width=1),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )

    # enforce categorical axis with the right order
    # fig.update_xaxes(
    #     type='category',
    #     categoryorder='array',
    #     categoryarray=ordered_sizes,
    #     tickangle=45  # tilt labels if needed
    # )

    fig.update_layout(
        # title="Iterations vs. Problem Size by Solver (Box-Plot)",
        template="plotly_white",
        # height=400,
        # width=310 * n_solvers,
        # margin=dict(t=60, b=50, l=50, r=50),
    )
    fig.update_yaxes(matches="y")
    return fig
