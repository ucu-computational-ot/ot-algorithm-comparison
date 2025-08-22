import dash
from dash import Output
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered

MAX_COLS = 3
ZMIN, ZMAX = 0.0, 1.0
LABEL_CELL_THRESHOLD = 56


@dash.callback(
    Output("desc-maxiter-heatmap", "figure"),   # <- add graph in layout with this id
    *FILTER_INPUTS,
)
def update_maxiter_heatmap(solvers, regs, dims, datasets, size):
    df = get_filtered(solvers, regs, dims, datasets, size)
    if df.empty or not {"solver","distribution","size","maxiter"} <= set(df.columns):
        fig = go.Figure()
        fig.update_layout(
            title="Max-iter hit rate per solver",
            annotations=[dict(text="No data for current filters", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    keys = ["distribution", "size", "solver"]
    colormap = "Cividis"

    # total runs
    total_runs = df.groupby(keys, dropna=False).size().rename("total_runs")

    # count runs where maxiter was hit
    maxiter_stats = (
        df.assign(is_maxiter=df["maxiter"] == df["iterations"])   # or ==1 if it's boolean
          .groupby(keys, dropna=False)["is_maxiter"]
          .sum()
          .rename("maxiter_count")
          .to_frame()
          .join(total_runs)
          .reset_index()
    )
    maxiter_stats["maxiter_rate"] = np.where(
        maxiter_stats["total_runs"] > 0,
        maxiter_stats["maxiter_count"] / maxiter_stats["total_runs"],
        np.nan,
    )

    # axis ordering
    row_labels = sorted(maxiter_stats["distribution"].unique(), key=str)
    def _num(v):
        try: return float(v)
        except Exception: return np.inf
    col_labels = sorted(maxiter_stats["size"].unique(), key=lambda x: (_num(x), str(x)))

    # facet grid
    solver_list = list(dict.fromkeys(maxiter_stats["solver"].tolist()))
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
        c = i % n_cols + 1

        d = maxiter_stats[maxiter_stats["solver"] == sol]

        M_rate = (
            d.pivot_table(index="distribution", columns="size",
                          values="maxiter_rate", aggfunc="mean")
             .reindex(index=row_labels)
             .reindex(columns=col_labels)
        )
        Z = M_rate.values.astype(float)
        text = (np.where(np.isnan(Z), "",
                         np.char.add((np.round(100*Z,1)).astype(str), "%"))
                if show_text else None)

        # counts for hover
        M_n = (
            d.pivot_table(index="distribution", columns="size",
                          values="total_runs", aggfunc="sum")
             .reindex(index=row_labels)
             .reindex(columns=col_labels)
        )

        fig.add_trace(
            go.Heatmap(
                z=Z,
                x=[str(x) for x in col_labels],
                y=[str(y) for y in row_labels],
                zmin=ZMIN, zmax=ZMAX,
                xgap=1,           # ← thin vertical borders between cells (in px)
                ygap=1,           # ← thin horizontal borders between cells
                colorscale=colormap,
                coloraxis="coloraxis",
                opacity=1,
                text=text,
                texttemplate="%{text}" if show_text else None,
                textfont={"size": 10},
                customdata=np.stack([M_n.values], axis=-1),
                hovertemplate=(
                    f"SOLVER: {sol}<br>"
                    "distribution: %{y}<br>"
                    "size: %{x}<br>"
                    "Max-iter hit rate: %{z:.3f}<br>"
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
                categoryarray=row_labels,     # fixed order
                tickmode="array",
                tickvals=row_labels,          # show EVERY category
                ticktext=row_labels,
                ticklabelposition="outside",
                automargin=True,              # prevent clipping
                row=r, col=c,
            )
            fig.update_yaxes(tickangle=30, tickfont=dict(size=12), row=r, col=c)

    # dynamic height
    base_per_row = 260
    per_cat = 18 * max(1, len(row_labels) - 6)
    height = max(320, n_rows * base_per_row + per_cat)

    fig.update_layout(
        coloraxis=dict(colorscale=colormap, cmin=ZMIN, cmax=ZMAX,
                       colorbar=dict(title="Max-iter hit rate", len=0.8)),
        margin=dict(l=70, r=20, t=60, b=80),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=16, color="black"),
        # font=dict(family="Times New Roman", size=16, color="black"),
    )

    fig.add_annotation(
        text="Number of points per axis",
        x=0.5, y=-0.06,
        xref="paper", yref="paper",
        xanchor="center", yanchor="top",
        showarrow=False,
        font=dict(family="Times New Roman", size=16, color="black"),
    )

    # fig.update_layout(plot_bgcolor="rgba(0,0,0,0.25)")  # subtle dark borders
    # or for white borders:
    fig.update_layout(plot_bgcolor="white")

    # fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)
    # fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5)

    # fig.update_xaxes(dtick=1, showgrid=True, gridcolor="lightgrey")
    # fig.update_yaxes(dtick=1, showgrid=True, gridcolor="lightgrey")

    # hide empty trailing subplots
    for k in range(n_sol, n_rows*n_cols):
        r = k // n_cols + 1
        c = k % n_cols + 1
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    return fig
