import dash
from dash import Output, dash_table
import pandas as pd
import numpy as np

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered


@dash.callback(
    Output("desc-summary-solver-distribution-table", "children"),
    *FILTER_INPUTS,
)
def update_summary_table(solvers, regs, dims, datasets, size):
    df = get_filtered(solvers, regs, dims, datasets, size, converged=True)
    required = {"solver", "distribution", "size", "runtime", "error", "cost_rerr"}
    if df.empty or not required.issubset(df.columns):
        return dash_table.DataTable(
            columns=[{"name": "Info", "id": "info"}],
            data=[{"info": "No data for current filters"}],
        )

    df = df.copy()
    for col in ["runtime", "error", "cost_rerr", "size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keys = ["solver", "distribution", "size"]

    summary = (
        df.groupby(keys)
          .agg(
              runtime_mean=("runtime", "mean"),
              runtime_std=("runtime", "std"),
              error_mean=("error", "mean"),
              error_std=("error", "std"),
              cost_rerr_mean=("cost_rerr", "mean"),
              cost_rerr_std=("cost_rerr", "std"),
          )
          .reset_index()
    )

    def fmt(m, s):
        if pd.isna(m): return "–"
        if pd.isna(s): return f"{m:.3g}"
        return f"{m:.3g} ± {s:.2g}"

    summary["runtime"] = [fmt(m,s) for m,s in zip(summary["runtime_mean"], summary["runtime_std"])]
    summary["error"]   = [fmt(m,s) for m,s in zip(summary["error_mean"], summary["error_std"])]
    summary["cost_rerr"] = [fmt(m,s) for m,s in zip(summary["cost_rerr_mean"], summary["cost_rerr_std"])]

    cols_out = ["solver", "distribution", "size", "runtime", "error", "cost_rerr"]

    return dash_table.DataTable(
        id={"type": "table", "target": "summary-runtime"},
        columns=[{"name": col, "id": col} for col in cols_out],
        data=summary[cols_out].to_dict("records"),
        # style_table={"overflowX": "auto", "maxHeight": "600px", "overflowY": "auto"},
        style_table={"overflowX": "auto", "overflowY": "auto"},
        style_cell={"textAlign": "left", "padding": "5px", "fontFamily": "monospace"},
        style_header={"fontWeight": "bold"},
    )
