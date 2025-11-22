import dash
from dash import Output
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered

# facet layout that wraps for many solvers
MAX_COLS = 3
ZMIN, ZMAX = 0.0, 1.0
LABEL_CELL_THRESHOLD = 56  # hide in-cell % labels when too dense


@dash.callback(
    Output("desc-instability-heatmap", "figure"),
    *FILTER_INPUTS,
)
def update_nan_heatmap(solvers, regs, dims, datasets, size):
    df = get_filtered(solvers, regs, dims, datasets, size)
    if df.empty or not {"solver","distribution","size","error"} <= set(df.columns):
        fig = go.Figure()
        fig.update_layout(
            title="NaN rate per solver",
            annotations=[dict(text="No data for current filters", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    # --- NaN-only statistics ---
    keys = ["distribution", "size", "solver"]
    colormap = "Cividis"

    # total rows (includes NaNs)
    total_runs = df.groupby(keys, dropna=False).size().rename("total_runs")

    # count NaNs in 'error' only
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

    # --- axes ordering ---
    row_labels = sorted(nan_stats["distribution"].unique(), key=str)
    def _num(v):
        try: return float(v)
        except Exception: return np.inf
    col_labels = sorted(nan_stats["size"].unique(), key=lambda x: (_num(x), str(x)))

    # --- facet grid that wraps for any number of solvers ---
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

        d = nan_stats[nan_stats["solver"] == sol]
        mat = (
            d.pivot_table(index="distribution", columns="size",
                          values="nan_rate", aggfunc="mean")
             .reindex(index=row_labels)
             .reindex(columns=col_labels)
        )
        Z = mat.values.astype(float)
        text = (np.where(np.isnan(Z), "", np.char.add((np.round(100*Z,1)).astype(str), "%"))
                if show_text else None)

        # we’ll also pass counts for hover
        Nmat = (
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
                colorscale=colormap,
                xgap=1,           # ← thin vertical borders between cells (in px)
                ygap=1,           # ← thin horizontal borders between cells
                coloraxis="coloraxis",   # shared colorbar
                text=text,
                texttemplate="%{text}" if show_text else None,
                textfont={"size": 10},
                customdata=np.stack([Nmat.values], axis=-1),
                hovertemplate=(
                    f"SOLVER: {sol}<br>"
                    "distribution: %{y}<br>"
                    "size: %{x}<br>"
                    "NaN rate: %{z:.3f}<br>"
                    "n: %{customdata[0]}<extra></extra>"
                ),
                hoverongaps=False,
            ),
            row=r, col=c,
        )

        # fig.update_xaxes(title_text="Size", row=r, col=c)
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
        # title="NaN rate per solver (share of runs with NaN error)",
        # coloraxis=dict(colorscale="Viridis", cmin=ZMIN, cmax=ZMAX,
        coloraxis=dict(colorscale=colormap, cmin=ZMIN, cmax=ZMAX,
                       colorbar=dict(title="NaN rate", len=0.8)),
        margin=dict(l=70, r=20, t=60, b=50),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=14),
    )
    # fig.update_layout(
    #     font=dict(family="Times New Roman", size=16, color="black"),
    #     title=dict(font=dict(size=14))
    # )
    fig.add_annotation(
        text="Number of point per axis",
        x=0.5, y=-0.03,         # по центру внизу
        xref="paper", yref="paper",
        xanchor="center", yanchor="top",
        showarrow=False,
        # font=dict(family="Times New Roman", size=16, color="black"),
        font=dict(size=16, color="black"),
    )

    # hide empty trailing subplots
    for k in range(n_sol, n_rows*n_cols):
        r = k // n_cols + 1
        c = k % n_cols + 1
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    return fig


