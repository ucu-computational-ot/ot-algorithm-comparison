import dash
from dash import Input, Output, html
from dash import dash_table
import numpy as np
import pandas as pd

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered


@dash.callback(
    Output("desc-size-growth-table-wrapper", "children"),   # container Div in layout
    *FILTER_INPUTS,
    # Input("desc-reg-runtime-only-converged", "value")
)
def render_size_growth_table(solvers, regs, dims, datasets, size):
# def render_size_growth_table(solvers, regs, dims, datasets, size, only_converged):
    # 1) Filter
    dff = get_filtered(solvers, regs, dims, datasets, size, converged=True).copy()

    # 2) Ensure numeric + valid
    dff["runtime"] = pd.to_numeric(dff["runtime"], errors="coerce")
    dff["size"]    = pd.to_numeric(dff["size"], errors="coerce")
    dff = dff.dropna(subset=["runtime", "size"])
    dff = dff[dff["runtime"] > 0]

    if dff.empty:
        return html.Div(
            "No data available for growth fitting by problem size.",
            style={"color": "#666", "fontStyle": "italic", "padding": "0.5rem 0"}
        )

    # 3) Aggregate: median runtime per (solver, size)
    med = (
        dff.groupby(["solver", "size"], as_index=False)
           .agg(median_runtime=("runtime", "median"))
    )

    rows = []
    for solver, g in med.groupby("solver"):
        g = g.sort_values("size")
        gs = g[g["size"] > 0].copy()  # avoid log(0)
        gs["log_runtime"] = np.log(gs["median_runtime"])

        # --- Power-law fit: log(runtime) ~ a0 + a1*log(size) ---
        if len(gs) >= 2:
            x1 = np.log(gs["size"].to_numpy())
            y1 = gs["log_runtime"].to_numpy()
            slope1, intercept1 = np.polyfit(x1, y1, 1)
            alpha = float(slope1)   # here slope is α_size
            y1_hat = intercept1 + slope1 * x1
            ss_res1 = float(np.sum((y1 - y1_hat) ** 2))
            ss_tot1 = float(np.sum((y1 - y1.mean()) ** 2))
            alpha_r2 = float(1 - ss_res1 / ss_tot1) if ss_tot1 > 0 else np.nan
            n_sizes_alpha = int(gs["size"].nunique())
        else:
            alpha, alpha_r2 = np.nan, np.nan
            n_sizes_alpha = int(gs["size"].nunique())

        # --- Exponential fit: log(runtime) ~ b0 + beta*size ---
        if len(gs) >= 2:
            x2 = gs["size"].to_numpy()
            y2 = gs["log_runtime"].to_numpy()
            slope2, intercept2 = np.polyfit(x2, y2, 1)
            beta = float(slope2)
            y2_hat = intercept2 + slope2 * x2
            ss_res2 = float(np.sum((y2 - y2_hat) ** 2))
            ss_tot2 = float(np.sum((y2 - y2.mean()) ** 2))
            beta_r2 = float(1 - ss_res2 / ss_tot2) if ss_tot2 > 0 else np.nan
            n_sizes_beta = int(gs["size"].nunique())
        else:
            beta, beta_r2 = np.nan, np.nan
            n_sizes_beta = int(gs["size"].nunique())

        rows.append({
            "solver": solver,
            "n_sizes": max(n_sizes_alpha, n_sizes_beta),
            "alpha_size": None if np.isnan(alpha) else round(alpha, 4),
            "alpha_r2": None if np.isnan(alpha_r2) else round(alpha_r2, 4),
            "beta_size": None if np.isnan(beta) else round(beta, 6),
            "beta_r2": None if np.isnan(beta_r2) else round(beta_r2, 4),
        })

    rows = sorted(rows, key=lambda r: r["solver"].lower())

    columns = [
        {"name": "Solver",             "id": "solver"},
        {"name": "#size levels",       "id": "n_sizes",    "type": "numeric"},
        {"name": "α_size (power-law)", "id": "alpha_size", "type": "numeric", "format": {"specifier": ".4f"}},
        {"name": "α_size R²",          "id": "alpha_r2",   "type": "numeric", "format": {"specifier": ".4f"}},
        {"name": "β_size (exponential)","id": "beta_size", "type": "numeric", "format": {"specifier": ".6f"}},
        {"name": "β_size R²",          "id": "beta_r2",    "type": "numeric", "format": {"specifier": ".4f"}},
    ]

    table = dash_table.DataTable(
        id="desc-size-growth-table",
        columns=columns,
        data=rows,
        sort_action="native",
        filter_action="native",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"padding": "8px", "fontFamily": "Inter, system-ui, sans-serif"},
        style_header={
            "fontWeight": "600",
            "backgroundColor": "#f7f7f9",
            "borderBottom": "1px solid #e6e6e6"
        },
        style_data={"borderBottom": "1px solid #f0f0f0"},
        style_data_conditional=[
            {"if": {"column_id": "solver"}, "textAlign": "left", "fontWeight": "500"},
            {"if": {"column_id": ["alpha_r2", "beta_r2"]}, "color": "#555"},
        ],
        export_format="csv",
        tooltip_header={
            "alpha_size": "Power-law: log(runtime) ~ a0 + α*log(size); α = slope",
            "beta_size":  "Exponential: log(runtime) ~ b0 + β*size; β = slope",
        },
        tooltip_duration=None,
    )

    return table
