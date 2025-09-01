import dash
from dash import Output, dash_table
import pandas as pd
import numpy as np

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered
from utils.perf_profile import _compute_ratios_per_run


GROUP_KEYS = ["dim", "solver", "distribution", "size"]
# GROUP_KEYS = ["dim", "solver", "reg", "distribution", "size"]

# @dash.callback(
#     Output("desc-normalized-runtime-summary-table", "children"),
#     *FILTER_INPUTS,
# )
# def update_normalized_runtime_table(solvers, regs, dims, datasets, size):
#     df = get_filtered(solvers, regs, dims, datasets, size, converged=True)
#     required = {"solver", "distribution", "size", "runtime", "reg", "dim"}
#     if df.empty or not required.issubset(df.columns):
#         return dash_table.DataTable(
#             columns=[{"name": "Info", "id": "info"}],
#             data=[{"info": "No data for current filters"}],
#         )
#
#     group = ["dim", "solver", "reg", "distribution", "size"]
#
#     df = df.copy()
#     df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
#
#     # define what a "problem instance" is
#     instance_keys = ["distribution", "size", "reg", "dim"]
#
#     # normalize per instance
#     normed_list = []
#     for keys, grp in df.groupby(instance_keys):
#         runtimes = grp.set_index("solver")["runtime"]
#         best = runtimes.min()
#         # skip degenerate cases
#         if best <= 0 or np.isnan(best):
#             continue
#         normed = runtimes / best
#         tmp = normed.reset_index()
#         for k, v in zip(instance_keys, keys):
#             tmp[k] = v
#         normed_list.append(tmp)
#
#     if not normed_list:
#         return dash_table.DataTable(
#             columns=[{"name": "Info", "id": "info"}],
#             data=[{"info": "No valid runtimes"}],
#         )
#
#     normed_df = pd.concat(normed_list, ignore_index=True)
#     normed_df.rename(columns={0: "norm_runtime"}, inplace=True)
#
#     # aggregate per solver
#     summary = (
#         # normed_df.groupby("solver")["runtime"]
#         normed_df.groupby(group)["runtime"]
#         .agg(
#             geom_mean=lambda x: np.exp(np.mean(np.log(x))),
#             median="median",
#             mean="mean",
#             std="std",
#         )
#         .reset_index()
#     )
#
#     # formatting
#     def fmt(x):
#         return f"{x:.3f}" if pd.notna(x) else "–"
#
#     for col in ["geom_mean", "median", "mean", "std"]:
#         summary[col] = summary[col].map(fmt)
#
#     return dash_table.DataTable(
#         id={"type": "table", "target": "normalized-runtime"},
#         columns=[{"name": col, "id": col} for col in summary.columns],
#         data=summary.to_dict("records"),
#         style_table={"overflowX": "auto"},
#         style_cell={"textAlign": "left", "padding": "5px", "fontFamily": "monospace"},
#         style_header={"fontWeight": "bold"},
#     )


@dash.callback(
    Output("desc-normalized-runtime-summary-table", "children"),
    *FILTER_INPUTS,
)
def update_normalized_runtime_table_runs(solvers, regs, dims, datasets, size):
    df = get_filtered(solvers, regs, dims, datasets, size, converged=True)
    need = {"solver","distribution","size","runtime","reg","dim"}
    if df.empty or not need.issubset(df.columns):
        return dash_table.DataTable(
            columns=[{"name": "Info", "id": "info"}],
            data=[{"info": "No data for current filters"}],
        )

    rt_col = "runtime" if "runtime" in df.columns else "time"

    # ratios with per-run alignment (introduces 'run_ix')
    ratios = _compute_ratios_per_run(df, runtime_col=rt_col)
    if ratios.empty:
        return dash_table.DataTable(
            columns=[{"name": "Info", "id": "info"}],
            data=[{"info": "No valid runs after normalization"}],
        )

    def gmean(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x) & (x > 0)]
        return float(np.exp(np.mean(np.log(x)))) if x.size else np.nan

    # Aggregate across runs (and instances) per group
    summary = (
        ratios.groupby(GROUP_KEYS, dropna=False)["ratio"]
              .agg(geom_mean=gmean, median="median", mean="mean", std="std")
              .reset_index()
    )

    def fmt(v): return f"{v:.3f}" if pd.notna(v) else "–"
    for c in ["geom_mean","median","mean","std"]:
        summary[c] = summary[c].map(fmt)

    return dash_table.DataTable(
        id={"type": "table", "target": "normalized-runtime"},
        columns=[{"name": c, "id": c} for c in summary.columns],
        data=summary.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px", "fontFamily": "monospace"},
        style_header={"fontWeight": "bold"},
    )
