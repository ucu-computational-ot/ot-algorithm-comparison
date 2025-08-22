import dash
from dash import Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered

# facet layout that wraps for many solvers
MAX_COLS = 3
ZMIN, ZMAX = 0.0, 1.0
LABEL_CELL_THRESHOLD = 56  # hide in-cell % labels when too dense
COLORMAP = "Cividis"


@dash.callback(
    Output("desc-instability-per-epsilon-heatmap", "figure"),   # same output id if you want to replace the old one
    *FILTER_INPUTS,
)
def update_nan_heatmap_vs_reg(solvers, regs, dims, datasets, size):
    df = get_filtered(solvers, regs, dims, datasets, size)
    # need these columns exactly as before, but using 'reg' instead of 'size'
    required = {"solver", "dataset", "reg", "error"}
    if df.empty or not required.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(
            title="NaN rate per solver vs regularization",
            annotations=[dict(text="No data for current filters", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    # coerce regularization and compute instability (NaN rate)
    df = df.copy()
    df["reg"] = pd.to_numeric(df["reg"], errors="coerce")
    df = df[~df["reg"].isna()]  # reg must be known on x

    keys = ["distribution", "reg", "solver"]

    total_runs = df.groupby(keys, dropna=False).size().rename("total_runs")
    nan_stats = (
        df.assign(is_nan_error=df["error"].isna())
          .groupby(keys, dropna=False)["is_nan_error"]
          .sum()
          .rename("nan_count")
          .to_frame()
          .join(total_runs)
          .reset_index()
    )
    nan_stats["nan_rate"] = np.where(
        nan_stats["total_runs"] > 0,
        nan_stats["nan_count"] / nan_stats["total_runs"],
        np.nan,
    )

    # axis ordering
    row_labels = sorted(nan_stats["distribution"].unique(), key=str)
    reg_vals = np.sort(nan_stats["reg"].dropna().unique())
    col_labels = [str(v) for v in reg_vals]  # keep labels as strings

    # facet grid
    solver_list = list(dict.fromkeys(nan_stats["solver"].tolist()))
    n_sol = len(solver_list)
    n_cols = min(MAX_COLS, max(1, n_sol))
    n_rows = int(np.ceil(n_sol / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.12 if n_rows > 1 else 0.08,
        subplot_titles=solver_list if solver_list else ["—"],
    )

    show_text = (len(col_labels) * len(row_labels) <= LABEL_CELL_THRESHOLD)

    for i, sol in enumerate(solver_list):
        r = i // n_cols + 1
        c = i %  n_cols + 1

        d = nan_stats[nan_stats["solver"] == sol].copy()
        d["reg_str"] = d["reg"].astype(str)

        M = (
            d.pivot_table(index="distribution", columns="reg_str",
                          values="nan_rate", aggfunc="mean")
             .reindex(index=row_labels)
             .reindex(columns=col_labels)
        )
        Z = M.values.astype(float)
        text = (np.where(np.isnan(Z), "", np.char.add((np.round(100*Z,1)).astype(str), "%"))
                if show_text else None)

        Nmat = (
            d.pivot_table(index="distribution", columns="reg_str",
                          values="total_runs", aggfunc="sum")
             .reindex(index=row_labels)
             .reindex(columns=col_labels)
        )

        fig.add_trace(
            go.Heatmap(
                z=Z,
                x=col_labels,
                y=[str(y) for y in row_labels],
                zmin=ZMIN, zmax=ZMAX,
                colorscale=COLORMAP,
                xgap=1, ygap=1,               # borders on cell edges
                coloraxis="coloraxis",        # shared colorbar
                text=text,
                texttemplate="%{text}" if show_text else None,
                textfont={"size": 10},
                customdata=np.stack([Nmat.values], axis=-1),
                hovertemplate=(
                    f"SOLVER: {sol}<br>"
                    "distribution: %{y}<br>"
                    "reg: %{x}<br>"
                    "NaN rate: %{z:.3f}<br>"
                    "n: %{customdata[0]}<extra></extra>"
                ),
                hoverongaps=False,
            ),
            row=r, col=c,
        )

        if c == 1:
            fig.update_yaxes(title_text="Distribution", row=r, col=c)
            fig.update_yaxes(
                type="category",
                categoryorder="array",
                categoryarray=row_labels,
                tickmode="array",
                tickvals=row_labels,
                ticktext=row_labels,
                ticklabelposition="outside",
                automargin=True,
                row=r, col=c,
            )
            fig.update_yaxes(tickangle=30, tickfont=dict(size=12), row=r, col=c)

        fig.update_xaxes(
            title_text="Regularization ε" if r == n_rows else None,
            categoryorder="array",
            categoryarray=col_labels,
            row=r, col=c
        )

    # dynamic height
    base_per_row = 260
    per_cat = 18 * max(1, len(row_labels) - 6)
    height = max(320, n_rows * base_per_row + per_cat)

    fig.update_layout(
        coloraxis=dict(colorscale=COLORMAP, cmin=ZMIN, cmax=ZMAX,
                       colorbar=dict(title="NaN rate", len=0.8)),
        margin=dict(l=70, r=20, t=60, b=50),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=14),
        # title="Instability (NaN rate) vs regularization (per solver)",
    )

    # fig.add_annotation(
    #     text="Regularization ε",
    #     x=0.5, y=-0.03,
    #     xref="paper", yref="paper",
    #     xanchor="center", yanchor="top",
    #     showarrow=False,
    #     font=dict(family="Times New Roman", size=16, color="black"),
    # )

    # hide empty trailing subplots
    for k in range(n_sol, n_rows*n_cols):
        r = k // n_cols + 1
        c = k %  n_cols + 1
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    return fig
