import dash
from dash import (
    html,
    dcc,
    Input,
    Output,
    dash_table,
)
from dash.dash_table.Format import Format, Scheme, Sign
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

from dataset import load_all_df, preprocess, filter_converged

# Load the full dataset once at module import
df_master = load_all_df()
preprocess(df_master)

# sorted lists
solvers   = sorted(df_master["solver"].unique())
regs      = sorted(df_master["reg"].dropna().astype(float).unique())
dims      = sorted(df_master["dim"].dropna().astype(int).unique())
datasets  = sorted(df_master["distribution"].unique())
npoints   = sorted(df_master["size"].dropna().astype(int).unique())

# checklist options
SOLVER_OPTIONS = [{"label": s, "value": s} for s in solvers]
REG_OPTIONS    = [{"label": r, "value": r} for r in regs]
DIM_OPTIONS    = [{"label": f"{d}D", "value": d} for d in dims]
DATASET_OPTIONS= [{"label": ds, "value": ds} for ds in datasets]
NPOINTS_OPTIONS= [{"label": n, "value": n} for n in npoints]

######################################################################
# Fix the color of each solver for every plot.
######################################################################
palette = px.colors.qualitative.G10
solvers_color_discrete_map = {
    solver: palette[i % len(palette)]
    for i, solver in enumerate(solvers)
}


# --- layout --------------------------------------------------------
dash.register_page(__name__, path="/descriptive", name="Descriptive")

layout = dbc.Container(fluid=True, class_name="p-4", children=[

    html.H2("Descriptive Statistics", className="mb-4"),

    # Filters
    dbc.Card(dbc.CardBody(
        dbc.Row([
            dbc.Col([
                html.Label("Solvers"),
                dbc.Checklist(
                    id="desc-solver-filter",
                    options=SOLVER_OPTIONS,
                    value=solvers,
                    switch=True,
                    persistence=True,
                    style={"maxHeight": "180px", "overflowY": "auto"}
                ),
            ], width=3),

            dbc.Col([
                html.Label("Regularization ε"),
                dbc.Checklist(
                    id="desc-reg-filter",
                    options=REG_OPTIONS,
                    value=regs,
                    switch=True,
                    persistence=True,
                    style={"maxHeight": "180px", "overflowY": "auto"}
                ),
            ], width=2),

            dbc.Col([
                html.Label("Dimension"),
                dbc.Checklist(
                    id="desc-dim-filter",
                    options=DIM_OPTIONS,
                    value=[],
                    # value=dims,
                    switch=True,
                    persistence=True
                ),
            ], width=2),

            dbc.Col([
                html.Label("Size (# points)"),
                dbc.Checklist(
                    id="desc-np-filter",
                    options=NPOINTS_OPTIONS,
                    value=npoints,
                    switch=True,
                    persistence=True
                ),
            ], width=2),

            dbc.Col([
                html.Label("Datasets"),
                dcc.Dropdown(
                    id="desc-ds-filter",
                    options=DATASET_OPTIONS,
                    value=datasets,
                    multi=True,
                    placeholder="Select datasets...",
                    persistence=True,
                ),
            ], width=3),
        ], class_name="g-3")
    ), class_name="mb-4"),

    # --- Section: Runtime Violin Plot ------------------------------
    html.H4("Runtime Violin Plot"),
    dcc.Graph(
        id='desc-runtime-violonplot',
        className="p-0"
    ),
    html.Hr(),

    # --- Section: Iteration vs Problem Size ------------------------
    html.H4("Iteration vs Problem Size"),
    dbc.Card([
        dbc.CardHeader("Instability Statistics"),
        dbc.CardBody(
            dbc.Table(id="desc-instability-stats",
                      striped=True,
                      bordered=True,
                      hover=True,
                      responsive=True)
        ),
    ]),
    dbc.Row([
        # dbc.Col(dbc.Table(id="desc-instability-stats", responsive=True), width=4),
        dbc.Col(dcc.Graph(id='desc-iteration-vs-problem-size'), width=8),
    ]),
    html.Hr(),

    # --- Section: Runtime Distribution -----------------------------
    html.H4("Runtime Distribution"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="desc-runtime-hist"), width=6),
        dbc.Col(dcc.Graph(id="desc-runtime-box"), width=6),
    ]),
    html.Hr(),

    # --- Section: Scaling & Error vs Runtime -----------------------
    html.H4("Scaling: Runtime vs Size"),
    dbc.Row([
        dbc.Checklist(
            id="desc-scaling-scatter-only-converged",
            options=[{"label": "", "value": True}],
            value=[],
            switch=True,
            persistence=True,
            inline=True,
        ),
        html.Label("Only Converged"),
    ],),
    dcc.Graph(id='desc-scaling-scatter'),
    html.H4("Error vs Runtime"),
    dcc.Graph(id='desc-error-runtime-scatter'),
    html.Hr(),

    # --- Section: Median Heatmap & Pivot Table ---------------------
    html.H4("Median Runtime Heatmap"),
    dbc.Row([
        dbc.Checklist(
            id="desc-median-pivot-only-converged",
            options=[{"label": "", "value": True}],
            value=[],
            switch=True,
            persistence=True,
            inline=True,
        ),
        html.Label("Only Converged"),
    ],),
    dcc.Graph(id="desc-median-heatmap"),
    html.H4("Pivot Table"),
    html.Div(id="desc-pivot-table", style={"overflowX": "auto"}),
    html.Hr(),

    # --- Section: Resource Usage -----------------------------------
    html.H4("Resource Usage"),
    dcc.Graph(id='desc-resources-violins'),


    html.H4("Cost Error"),
    dcc.Graph(id='desc-cost-error-vs-runtime'),

    dcc.Graph(id='desc-cost-err-problem-size-violonplot'),

    dcc.Graph(id='desc-cost-rerr-vs-problem-size'),
])


# Helper: filter master df
def _filter_df(solvers, regs, dims, datasets, size):
    dff = df_master[
        df_master["solver"].isin(solvers) &
        df_master["reg"].isin(regs) &
        df_master["dim"].isin(dims) &
        df_master["distribution"].isin(datasets) &
        df_master["size"].isin(size)
    ]
    return dff


# Callbacks
dash.register_page(__name__, path="/descriptive")

########################################################
# ---------------- RUNTIME VIOLONPLOT ------------------
########################################################
@dash.callback(
    Output('desc-runtime-violonplot', "figure"),
    Input("desc-solver-filter",    "value"),
    Input("desc-reg-filter",       "value"),
    Input("desc-dim-filter",       "value"),
    Input("desc-ds-filter",        "value"),
    Input("desc-np-filter",        "value"),
)
def update_runtime_violonplot(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)
    dff = filter_converged(dff)
    fig = px.violin(
        dff,
        x="runtime",
        y="dataset",
        color="solver",
        orientation="h",
        box=True,
        points="all",
        hover_data=["iterations", "reg"],
        color_discrete_map=solvers_color_discrete_map,
    )
    fig.update_traces(
        width=0.8,
    )
    N = len(dff['dataset'].unique())
    per_row = 50
    base = 200
    fig.update_layout(
        height=base + per_row * N,
        margin=dict(l=200, r=20, t=50, b=50),
        title="Solver Timings Across All Datasets",
        xaxis_title="Time (s)",
        yaxis_title="Dataset",
        legend_title="Solver",
        template="plotly_white",
    )
    return fig

################################################################################
# ------------------ ITERATIONS VS SIZE BOX‐PLOT SUBPLOTS ---------------------
################################################################################
@dash.callback(
    Output("desc-iteration-vs-problem-size", "figure"),
    Input("desc-solver-filter", "value"),
    Input("desc-reg-filter",    "value"),
    Input("desc-dim-filter",    "value"),
    Input("desc-ds-filter",     "value"),
    Input("desc-np-filter",     "value"),
)
def update_iterations_boxplot(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)
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
        color = solvers_color_discrete_map.get(solver, "#444")

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
    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=ordered_sizes,
        tickangle=45  # tilt labels if needed
    )

    fig.update_layout(
        title="Iterations vs. Problem Size by Solver (Box-Plot)",
        template="plotly_white",
        height=400,
        width=300 * n_solvers,
        margin=dict(t=60, b=50, l=50, r=50),
    )
    fig.update_yaxes(matches="y")
    return fig

################################################################################
# ------------------ RUNTIME STATISTICS ---------------------
################################################################################
@dash.callback(
    Output("desc-runtime-hist", "figure"),
    Input("desc-solver-filter",    "value"),
    Input("desc-reg-filter",       "value"),
    Input("desc-dim-filter",       "value"),
    Input("desc-ds-filter",        "value"),
    Input("desc-np-filter",        "value"),
)
def update_runtime_hist(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)
    fig = px.histogram(
        dff,
        x="runtime",
        color="solver",
        marginal="box",
        nbins=50,
        opacity=0.7,
        title="Runtime Distribution",
        color_discrete_map=solvers_color_discrete_map,
    )
    fig.update_layout(barmode="overlay")
    return fig


@dash.callback(
    Output("desc-runtime-box", "figure"),
    Input("desc-solver-filter",    "value"),
    Input("desc-reg-filter",       "value"),
    Input("desc-dim-filter",       "value"),
    Input("desc-ds-filter",        "value"),
    Input("desc-np-filter",        "value"),
)
def update_runtime_box(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)
    fig = px.box(
        dff, x="solver", y="runtime", color="solver",
        title="Runtime by Solver",
        color_discrete_map=solvers_color_discrete_map,
    )
    return fig

################################################################################
# ------------------ SCALING STATISTICS---------------------
################################################################################
@dash.callback(
    Output("desc-scaling-scatter", "figure"),
    Input("desc-solver-filter",    "value"),
    Input("desc-reg-filter",       "value"),
    Input("desc-dim-filter",       "value"),
    Input("desc-ds-filter",        "value"),
    Input("desc-np-filter",        "value"),
    Input("desc-scaling-scatter-only-converged", "value"),
)
def update_scaling_scatter(solvers, regs, dims, datasets, size, only_converged):
    # 1) filter & aggregate
    dff = _filter_df(solvers, regs, dims, datasets, size)
    if only_converged:
        dff = filter_converged(dff)
    agg = (
        dff.groupby(["solver", "size"], as_index=False)
           .agg(mean_runtime=("runtime", "mean"))
    )

    # 2) line plot with markers
    fig = px.line(
        agg.sort_values("size"),
        x="size",
        y="mean_runtime",
        color="solver",
        markers=True,
        title="Mean Runtime vs Problem Size",
        labels={
            "size": "Problem Size (# points)",
            "mean_runtime": "Mean Runtime (s)"
        },
        color_discrete_map=solvers_color_discrete_map,
    )

    # 3) log–log axes
    fig.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=sorted(agg["size"].unique()),
        ticktext=[str(int(v)) for v in sorted(agg["size"].unique())],
        ticks="outside",
        tickangle=45
    )
    fig.update_yaxes(type="log", ticks="outside")

    # 4) tighten layout
    fig.update_layout(
        template="plotly_white",
        legend_title="Solver",
        margin=dict(l=60, r=20, t=50, b=60),
        height=450,
    )

    return fig


@dash.callback(
    Output("desc-error-runtime-scatter", "figure"),
    Input("desc-solver-filter",    "value"),
    Input("desc-reg-filter",       "value"),
    Input("desc-dim-filter",       "value"),
    Input("desc-ds-filter",        "value"),
    Input("desc-np-filter",        "value"),
)
def update_error_runtime(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)
    # fig = px.scatter(
    #     dff, x="error", y="runtime", color="solver", symbol="dim",
    #     title="Error vs Runtime"
    # )
    fig = px.scatter(
        dff,
        x="runtime",
        y="error",
        color="solver",
        color_discrete_map=solvers_color_discrete_map,
        symbol="dim",
        title="Error vs Runtime",
        marginal_x="histogram",
        marginal_y="violin",
        log_x=True, log_y=True,
        template="plotly_white"
    )
    fig.add_hline(y=1e-6, line_dash="dash", line_color="gray")
    fig.update_xaxes(
        tickformat=".1e",             # e.g. “2.0e-05” instead of “20 µ”
        exponentformat="e",
        showexponent="all",
        # type="log",
    )
    fig.update_yaxes(
        tickformat=".1e",             # e.g. “2.0e-05” instead of “20 µ”
        exponentformat="e",
        showexponent="all",
        # type="log",
    )
    return fig


@dash.callback(
    Output("desc-median-heatmap", "figure"),
    Input("desc-solver-filter",    "value"),
    Input("desc-reg-filter",       "value"),
    Input("desc-dim-filter",       "value"),
    Input("desc-ds-filter",        "value"),
    Input("desc-np-filter",        "value"),
)
def update_median_heatmap(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)
    pivot = (
        dff.groupby(["dim", "solver"])["runtime"]
           .median()
           .reset_index()
           .pivot(index="dim", columns="solver", values="runtime")
    )
    fig = px.imshow(
        pivot,
        labels={"x": "Solver", "y": "Dimension", "color": "Median runtime"},
        title="Median Runtime Heatmap",
        aspect="auto",
    )
    return fig


@dash.callback(
    Output("desc-pivot-table", "children"),
    Input("desc-solver-filter", "value"),
    Input("desc-reg-filter",    "value"),
    Input("desc-dim-filter",    "value"),
    Input("desc-ds-filter",     "value"),
    Input("desc-np-filter",     "value"),
    Input("desc-median-pivot-only-converged",     "value"),
)
def update_pivot_table(solvers, regs, dims, datasets, size, only_converged):
    dff = _filter_df(solvers, regs, dims, datasets, size)
    if only_converged:
        dff = filter_converged(dff)
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


@dash.callback(
    Output("desc-instability-stats", "children"),
    Input("desc-solver-filter", "value"),
    Input("desc-reg-filter",    "value"),
    Input("desc-dim-filter",    "value"),
    Input("desc-ds-filter",     "value"),
    Input("desc-np-filter",     "value"),
)
def update_instability_stats(solvers, regs, dims, datasets, size):
    # 1) Pull & filter the master DF
    total = _filter_df(solvers, regs, dims, datasets, size)

    # 2) Drop rows with NaN error from the “converged” set
    converged = filter_converged(total)

    # 3) Group total runs
    grp_total = total.groupby(["dim", "solver"]).agg(
        nan_count      = ("error",      lambda x: x.isna().sum()),
        max_iterations = ("iterations", "max"),
        total_runs     = ("error",      "count"),
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

###############################################################
# -------------- RESOURCE USAGE VIOLIN PLOT CALLBACK ----------
###############################################################
@dash.callback(
    Output("desc-resources-violons", "figure"),
    Input("desc-solver-filter", "value"),
    Input("desc-reg-filter",    "value"),
    Input("desc-dim-filter",    "value"),
    Input("desc-ds-filter",     "value"),
    Input("desc-np-filter",     "value"),
)
def update_resource_violinplot(solvers, regs, dims, datasets, size):
    # 1) filter the main DataFrame
    dff = _filter_df(solvers, regs, dims, datasets, size)

    # 2) melt memory metrics, then drop zero readings
    mem = (
        dff.melt(
            id_vars=["dataset", "solver"],
            value_vars=["peak_gpu_mem", "combined_peak_gpu_mem"],
            var_name="metric",
            value_name="value",
        )
        .assign(metric_type="Memory (MiB)")
        .query("value > 0")  # drop zeros only in memory facet
    )

    # 3) melt utilization metrics, then drop zeros
    util = (
        dff.melt(
            id_vars=["dataset", "solver"],
            value_vars=["mean_util_pct"],
            # value_vars=["peak_util_pct", "mean_util_pct"],
            var_name="metric",
            value_name="value",
        )
        .assign(metric_type="Utilization (%)")
        .query("value > 0")  # drop zeros only in utilization facet
    )

    # 4) combine
    long = pd.concat([mem, util], ignore_index=True)
    print(long)

    # 5) if nothing remains, show placeholder
    if long.empty:
        return {
            "data": [],
            "layout": {"title": "No resource data available for these filters"}
        }

    # 6) plot
    fig = px.violin(
        long,
        x="value",
        y="dataset",
        color="solver",
        color_discrete_map=solvers_color_discrete_map,
        facet_col="metric_type",
        orientation="h",
        box=True,
        points="all",
        hover_data=["metric"],
        category_orders={"metric_type": ["Memory (MiB)", "Utilization (%)"]},
        template="plotly_white",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig.update_layout(
        title="Resource Usage by Dataset and Solver",
        legend_title="Solver",
        height=500,
        width=900,
        margin=dict(t=60, b=50, l=200, r=50),
    )
    fig.update_xaxes(title_text="Memory (MiB)", matches=None, col=1)
    fig.update_xaxes(title_text="Utilization (%)", matches=None, col=2)
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash", zeroline=False)

    return fig


@dash.callback(
    Output("desc-resources-violins", "figure"),
    Input("desc-solver-filter", "value"),
    Input("desc-reg-filter",    "value"),
    Input("desc-dim-filter",    "value"),
    Input("desc-ds-filter",     "value"),
    Input("desc-np-filter",     "value"),
)
def update_resource_violinplot(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)

    # drop rows where ALL metrics are zero, as before
    metrics_to_check = [
        "peak_gpu_mem", "combined_peak_gpu_mem",
        "peak_ram_mem", "combined_peak_ram_mem",
        "peak_util_pct", "mean_util_pct",
        "peak_cpu_util_pct", "mean_cpu_util_pct"
    ]
    dff = dff[~(dff[metrics_to_check] == 0).all(axis=1)]

    # melt each group
    def melt_group(df, vars, metric_type):
        return (
            df.melt(
                id_vars=["dataset", "solver"],
                value_vars=vars,
                var_name="metric",
                value_name="value",
            )
            .assign(metric_type=metric_type)
            .query("value > 0")
        )

    mem_gpu = melt_group(dff, ["peak_gpu_mem", "combined_peak_gpu_mem"], "GPU Memory (MiB)")
    mem_cpu = melt_group(dff, ["peak_ram_mem",   "combined_peak_ram_mem"], "CPU Memory (MiB)")
    util_gpu = melt_group(dff, ["peak_util_pct",  "mean_util_pct"],        "GPU Utilization (%)")
    util_cpu = melt_group(dff, ["peak_cpu_util_pct","mean_cpu_util_pct"],   "CPU Utilization (%)")

    long = pd.concat([mem_gpu, mem_cpu, util_gpu, util_cpu], ignore_index=True)

    if long.empty:
        return {
            "data": [], 
            "layout": {"title": "No resource data available for these filters"}
        }

    # 2×2 facet: rows=Memory/Util, cols=GPU/CPU
    fig = px.violin(
        long,
        x="value",
        y="dataset",
        color="solver",
        color_discrete_map=solvers_color_discrete_map,
        facet_row="metric_type",       # this will produce 4 rows if you omit facet_col
        orientation="h",
        box=True,
        points="all",
        hover_data=["metric"],
        category_orders={
          "metric_type": [
              "GPU Memory (MiB)", "CPU Memory (MiB)",
              "GPU Utilization (%)", "CPU Utilization (%)"
          ]
        },
        template="plotly_white",
    )

    # Tidy up the facet labels
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    fig.update_layout(
        title="Resource Usage by Dataset and Solver",
        legend_title="Solver",
        height=800,
        margin=dict(t=80, b=50, l=200, r=50),
    )

    # Individual axis titles
    fig.update_xaxes(matches=None, title_text="MiB", row=1, col=1)
    fig.update_xaxes(matches=None, title_text="MiB", row=2, col=1)
    fig.update_xaxes(matches=None, title_text="Percent", row=3, col=1)
    fig.update_xaxes(matches=None, title_text="Percent", row=4, col=1)

    # grid styling
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash", zeroline=False)

    return fig


@dash.callback(
    Output("desc-cost-error-vs-runtime", "figure"),
    Input("desc-solver-filter", "value"),
    Input("desc-reg-filter",    "value"),
    Input("desc-dim-filter",    "value"),
    Input("desc-ds-filter",     "value"),
    Input("desc-np-filter",     "value"),
)
def update_cost_error_vs_runtime_plot(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)

    fig = px.scatter(
        dff,
        # x="runtime",
        # y="cost_rerr",

        y="runtime",
        x="cost_rerr",

        color="solver",
        color_discrete_map=solvers_color_discrete_map,
        symbol="dim",
        title="Cost Relative Error vs Runtime",
        marginal_x="histogram",
        marginal_y="violin",
        log_x=True,
        # log_y=True,
        template="plotly_white"
    )
    # fig.add_hline(y=1e-6, line_dash="dash", line_color="gray")
    fig.update_xaxes(
        tickformat=".1e",             # e.g. “2.0e-05” instead of “20 µ”
        exponentformat="e",
        showexponent="all",
        # type="log",
    )
    fig.update_yaxes(
        tickformat=".1e",             # e.g. “2.0e-05” instead of “20 µ”
        exponentformat="e",
        showexponent="all",
        # type="log",
    )
    return fig


@dash.callback(
    Output('desc-cost-err-problem-size-violonplot', "figure"),
    Input("desc-solver-filter",    "value"),
    Input("desc-reg-filter",       "value"),
    Input("desc-dim-filter",       "value"),
    Input("desc-ds-filter",        "value"),
    Input("desc-np-filter",        "value"),
)
def update_cost_rerr_violinplot(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)
    dff = filter_converged(dff)
    fig = px.violin(
        dff,
        x="cost_rerr",
        y="dataset",
        color="solver",
        orientation="h",
        box=True,
        points="all",
        hover_data=["iterations", "reg", "solver"],
        color_discrete_map=solvers_color_discrete_map,
    )
    fig.update_traces(
        width=0.8,
    )
    N = len(dff['dataset'].unique())
    per_row = 50
    base = 200
    fig.update_layout(
        height=base + per_row * N,
        margin=dict(l=200, r=20, t=50, b=50),
        title="Solver Cost RERR Across All Datasets",
        xaxis_title="Cost Relative Error",
        yaxis_title="Dataset",
        legend_title="Solver",
        template="plotly_white",
    )
    return fig


@dash.callback(
    Output("desc-cost-rerr-vs-problem-size", "figure"),
    Input("desc-solver-filter", "value"),
    Input("desc-reg-filter",    "value"),
    Input("desc-dim-filter",    "value"),
    Input("desc-ds-filter",     "value"),
    Input("desc-np-filter",     "value"),
)
def update_cost_error_vs_size_boxplot(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)
    n_solvers = len(solvers)

    # precompute the ordered categories as strings
    ordered_sizes = [str(n) for n in sorted(dff["size"].astype(int).unique())]

    fig = make_subplots(
        rows=1,
        cols=n_solvers,
        subplot_titles=solvers,
        shared_yaxes=True,
        x_title="Problem Size (# points)",
        y_title="Cost RERR",
    )

    for col_idx, solver in enumerate(solvers, start=1):
        subset = dff[dff["solver"] == solver]
        # convert to string so it's treated categorically
        x_cat = subset["size"].astype(int).astype(str)
        color = solvers_color_discrete_map.get(solver, "#444")

        fig.add_trace(
            go.Box(
                x=x_cat,
                y=subset["cost_rerr"],
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
    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=ordered_sizes,
        tickangle=45  # tilt labels if needed
    )

    fig.update_layout(
        title="Cost RERR vs. Problem Size by Solver (Box-Plot)",
        template="plotly_white",
        height=400,
        width=300 * n_solvers,
        margin=dict(t=60, b=50, l=50, r=50),
    )
    fig.update_yaxes(matches="y")
    return fig
