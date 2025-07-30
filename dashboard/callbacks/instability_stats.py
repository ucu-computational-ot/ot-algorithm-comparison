import dash
from dash import Output, dash_table

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered
from dataset import filter_converged


@dash.callback(
    Output("desc-instability-stats", "children"),
    *FILTER_INPUTS,
)
def update_instability_stats(solvers, regs, dims, datasets, size):
    # 1) Pull & filter the master DF
    total = get_filtered(solvers, regs, dims, datasets, size)

    # 2) Drop rows with NaN error from the “converged” set
    converged = filter_converged(total)

    # 3) Group total runs
    grp_total = total.groupby(["dim", "solver"]).agg(
        nan_count = ("error", lambda x: x.isna().sum()),
        max_iterations = ("iterations", "max"),
        total_runs = ("error", "count"),
    )

    # 4) Count converged runs
    grp_conv = converged.groupby(["dim", "solver"]).size().rename("converged_runs")

    # 5) Join and compute failures
    summary = (
        grp_total
        .join(grp_conv, how="left")
        .fillna({"converged_runs": 0})
        .assign(
            failure_count=lambda df: df["total_runs"] - df["converged_runs"]
        )
        .reset_index()
        # drop the helper columns if you like
        .drop(columns=["total_runs", "converged_runs"])
    )
# 6) Return as a DataTable
    return dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in summary.columns],
        data=summary.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px"},
    )
