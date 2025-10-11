import dash
from dash import Input, Output, html
from dash import dash_table
import numpy as np
import pandas as pd

from utils.filters import FILTER_INPUTS          # (solvers, regs, dims, datasets, size)
from utils.data import get_filtered


@dash.callback(
    Output("desc-reg-growth-table-wrapper", "children"),   # <- container Div in layout
    *FILTER_INPUTS,
    # Input("desc-reg-runtime-only-converged", "value")
)
def render_growth_table(solvers, regs, dims, datasets, size):
# def render_growth_table(solvers, regs, dims, datasets, size, only_converged):
    # 1) Filter
    dff = get_filtered(solvers, regs, dims, datasets, size, converged=True).copy()

    # 2) Ensure numeric + valid
    dff["runtime"] = pd.to_numeric(dff["runtime"], errors="coerce")
    dff["reg"]     = pd.to_numeric(dff["reg"], errors="coerce")
    dff = dff.dropna(subset=["runtime", "reg"])
    dff = dff[dff["runtime"] > 0]

    # Empty state
    if dff.empty:
        return html.Div(
            "No data available for growth fitting after filters.",
            style={"color": "#666", "fontStyle": "italic", "padding": "0.5rem 0"}
        )

    # 3) Robust aggregate: median runtime per (solver, ε)
    med = (
        dff.groupby(["solver", "reg"], as_index=False)
           .agg(median_runtime=("runtime", "median"))
    )

    # 4) Fit α (power-law) and β (exponential) per solver
    rows = []
    for solver, g in med.groupby("solver"):
        g = g.sort_values("reg")
        gp = g[g["reg"] > 0].copy()  # power-law needs ε>0
        ge = g.copy()

        gp["log_runtime"] = np.log(gp["median_runtime"])
        ge["log_runtime"] = np.log(ge["median_runtime"])

        # Fit 1: log(runtime) ~ a0 + a1*log(eps)  => alpha = -a1
        if len(gp) >= 2:
            x1 = np.log(gp["reg"].to_numpy())
            y1 = gp["log_runtime"].to_numpy()
            slope1, intercept1 = np.polyfit(x1, y1, 1)
            alpha = float(-slope1)
            y1_hat = intercept1 + slope1 * x1
            ss_res1 = float(np.sum((y1 - y1_hat) ** 2))
            ss_tot1 = float(np.sum((y1 - y1.mean()) ** 2))
            alpha_r2 = float(1 - ss_res1 / ss_tot1) if ss_tot1 > 0 else np.nan
            n_eps_alpha = int(gp["reg"].nunique())
        else:
            alpha, alpha_r2 = np.nan, np.nan
            n_eps_alpha = int(gp["reg"].nunique())

        # Fit 2: log(runtime) ~ b0 + beta*eps    => beta = slope2
        if len(ge) >= 2:
            x2 = ge["reg"].to_numpy()
            y2 = ge["log_runtime"].to_numpy()
            slope2, intercept2 = np.polyfit(x2, y2, 1)
            beta = float(slope2)
            y2_hat = intercept2 + slope2 * x2
            ss_res2 = float(np.sum((y2 - y2_hat) ** 2))
            ss_tot2 = float(np.sum((y2 - y2.mean()) ** 2))
            beta_r2 = float(1 - ss_res2 / ss_tot2) if ss_tot2 > 0 else np.nan
            n_eps_beta = int(ge["reg"].nunique())
        else:
            beta, beta_r2 = np.nan, np.nan
            n_eps_beta = int(ge["reg"].nunique())

        rows.append({
            "solver": solver,
            "n_eps": max(n_eps_alpha, n_eps_beta),
            "alpha": None if np.isnan(alpha) else round(alpha, 4),
            "alpha_r2": None if np.isnan(alpha_r2) else round(alpha_r2, 4),
            "beta": None if np.isnan(beta) else round(beta, 6),
            "beta_r2": None if np.isnan(beta_r2) else round(beta_r2, 4),
        })

    rows = sorted(rows, key=lambda r: r["solver"].lower())

    columns = [
        {"name": "Solver",             "id": "solver"},
        {"name": "#ε levels",          "id": "n_eps",    "type": "numeric"},
        {"name": "alpha (power-law)",  "id": "alpha",    "type": "numeric", "format": {"specifier": ".4f"}},
        {"name": "alpha R²",           "id": "alpha_r2", "type": "numeric", "format": {"specifier": ".4f"}},
        {"name": "beta (exponential)", "id": "beta",     "type": "numeric", "format": {"specifier": ".6f"}},
        {"name": "beta R²",            "id": "beta_r2",  "type": "numeric", "format": {"specifier": ".4f"}},
    ]

    table = dash_table.DataTable(
        id="desc-reg-growth-table",
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
            "alpha": "Power-law: log(runtime) ~ a0 + a1*log(ε); alpha = -a1",
            "beta":  "Exponential: log(runtime) ~ b0 + beta*ε; beta = slope",
        },
        tooltip_duration=None,
    )

    return table
