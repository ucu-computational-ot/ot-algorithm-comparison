import dash
from dash import Input, Output, dash_table
from dash.dash_table.Format import Format, Scheme, Sign

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered


@dash.callback(
    Output("desc-pivot-table", "children"),
    *FILTER_INPUTS,
    Input("desc-median-pivot-only-converged", "value"),
)
def update_pivot_table(solvers, regs, dims, datasets, size, only_converged):
    dff = get_filtered(solvers, regs, dims, datasets, size, only_converged)
    summary = (
        dff.groupby(["dim", "solver"])
           .agg(
               mean_runtime=("runtime", "mean"),
               std_runtime=("runtime", "std"),
               mean_error=("error", "mean"),
               std_error=("error", "std"),
           )
           # keep full precision here
           .reset_index()
    )

    # Build column definitions with formats
    columns = [
        {"name": "Dimension",      "id": "dim",           "type": "numeric"},
        {"name": "Solver",         "id": "solver",        "type": "text"},
        {"name": "Mean Runtime",   "id": "mean_runtime",  "type": "numeric",
         "format": Format(precision=4, scheme=Scheme.fixed,   sign=Sign.positive)},
        {"name": "Std Runtime",    "id": "std_runtime",   "type": "numeric",
         "format": Format(precision=4, scheme=Scheme.fixed,   sign=Sign.positive)},
        {"name": "Mean Error",     "id": "mean_error",    "type": "numeric",
         "format": Format(precision=2, scheme=Scheme.exponent, sign=Sign.positive)},
        {"name": "Std Error",      "id": "std_error",     "type": "numeric",
         "format": Format(precision=2, scheme=Scheme.exponent, sign=Sign.positive)},
    ]

    return dash_table.DataTable(
        columns=columns,
        data=summary.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px"},
        page_size=10,
    )
