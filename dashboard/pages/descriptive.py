import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

from dataset import load_all_df, preprocess

# Load the full dataset once at module import
df_master = load_all_df()
preprocess(df_master)

# Precompute unique options for filters
SOLVER_OPTIONS = [
    {"label": s, "value": s}
    for s in sorted(df_master["solver"].unique())
]
REG_OPTIONS = [
    {"label": str(r), "value": r}
    for r in sorted(df_master["reg"].unique(), key=lambda x: (x is None, x))
]
DIM_OPTIONS = [
    {"label": f"{d}D", "value": d}
    for d in sorted(df_master["dim"].unique())
]
DATASET_OPTIONS = [
    {"label": ds, "value": ds}
    for ds in sorted(df_master["distribution"].unique())
]
NPOINTS_OPTIONS = [
    {"label": str(n), "value": n} for n in sorted(df_master["size"].unique())
]

layout = dbc.Container([
    html.H2("Descriptive Statistics", className="my-4"),

    ###################################################################################
    # ------------------------- FILTER CONTROLS -----------------------------
    ###################################################################################
    html.Div([
        html.Div([
            html.Label("Solver"),
            dcc.Dropdown(
                id="desc-solver-filter",
                options=SOLVER_OPTIONS,
                multi=True,
                value=[o["value"] for o in SOLVER_OPTIONS],
            ),
        ], style={"width": "20%", "display": "inline-block", "padding": "0 10px"}),

        html.Div([
            html.Label("Regularization ε"),
            dcc.Dropdown(
                id="desc-reg-filter",
                options=REG_OPTIONS,
                multi=True,
                value=[o["value"] for o in REG_OPTIONS],
            ),
        ], style={"width": "20%", "display": "inline-block", "padding": "0 10px"}),

        html.Div([
            html.Label("Dimension"),
            dcc.Dropdown(
                id="desc-dim-filter",
                options=DIM_OPTIONS,
                multi=True,
                value=[o["value"] for o in DIM_OPTIONS],
            ),
        ], style={"width": "15%", "display": "inline-block", "padding": "0 10px"}),

        html.Div([
            html.Label("Dataset"),
            dcc.Dropdown(
                id="desc-ds-filter",
                options=DATASET_OPTIONS,
                multi=True,
                value=[o["value"] for o in DATASET_OPTIONS],
            ),
        ], style={"width": "25%", "display": "inline-block", "padding": "0 10px"}),

        html.Div([
            html.Label("size"),
            dcc.Dropdown(
                id="desc-np-filter",
                options=NPOINTS_OPTIONS,
                multi=True,
                value=[o["value"] for o in NPOINTS_OPTIONS],
            ),
        ], style={"width": "15%", "display": "inline-block", "padding": "0 10px"}),
    ], style={"margin-bottom": "25px"}),

    ###################################################################################
    # ------------------------- GRAPHS GRID -----------------------------
    ###################################################################################

    # --------------------- RUNTIME DISTRIBUTION ----------------------
    html.Div([
        html.Div([
            html.H4("Runtime ViolonPlot"),
            dcc.Graph(id='desc-runtime-violonplot'),
        ], className="six columns")
    ], className="row"),
    # --------------------- ITERATION VS SIZE ----------------------
    html.Div([
        html.Div(id="desc-instability-stats", style={"overflowX": "auto"}),
        html.Div([
            html.H4("Iteration VS Problem Size"),
            dcc.Graph(id='desc-iteration-vs-problem-size'),
        ], className="six columns")
    ], className="row"),

    # --------------------- RUNTIME DISTRIBUTION ----------------------
    html.Div([
        html.Div([
            html.H4("Runtime Distribution"),
            dcc.Graph(id="desc-runtime-hist"),
        ], className="six columns"),

        html.Div([
            html.H4("Runtime by Solver (Box Plot)"),
            dcc.Graph(id="desc-runtime-box"),
        ], className="six columns"),
    ], className="row"),

    html.Div([
        html.Div([
            html.H4("Scaling: Runtime vs Size"),
            dcc.Graph(id="desc-scaling-scatter"),
        ], className="six columns"),

        html.Div([
            html.H4("Error vs Runtime"),
            dcc.Graph(id="desc-error-runtime-scatter"),
        ], className="six columns"),
    ], className="row"),

    html.Div([
        html.Div([
            html.H4("Median Runtime Heatmap"),
            dcc.Graph(id="desc-median-heatmap"),
        ], className="six columns"),

        html.Div([
            html.H4("Pivot Table"),
            html.Div(id="desc-pivot-table", style={"overflowX": "auto"}),
        ], className="six columns"),
    ], className="row"),
    # --------------------- RESOURCES ----------------------
    html.Div([
        html.Div([
            html.H4("Resources"),
            dcc.Graph(id='desc-resources-violon'),
        ], className="six columns")
    ], className="row"),
],
    fluid=True,
    className="px-0",
    )


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
    fig = px.violin(
        dff,
        x="runtime",
        y="dataset",
        color="solver",
        orientation="h",
        box=True,
        # points="all",
        hover_data=["iterations", "reg"],
        # color_discrete_map=color_map,  # force each solver to its color
    )
    fig.update_traces(
        width=0.8,  
    )
    N = len(dff['dataset'].unique())  
    per_row = 20
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
    print(n_solvers)
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
        fig.add_trace(
            go.Box(
                x=subset["size"],
                y=subset["iterations"],
                name=solver,
                boxpoints="all",
                jitter=0.3,
                marker=dict(size=4, opacity=0.6),
                line=dict(width=1),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
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
        dff, x="runtime", color="solver", marginal="box",
        nbins=50, title="Runtime Distribution"
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
        title="Runtime by Solver"
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
)
def update_scaling_scatter(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)
    agg = dff.groupby(["solver", "size"])["runtime"].mean().reset_index()
    fig = px.scatter(
        agg, x="size", y="runtime", color="solver", trendline="ols",
        title="Mean Runtime vs size"
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
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
    Input("desc-solver-filter",    "value"),
    Input("desc-reg-filter",       "value"),
    Input("desc-dim-filter",       "value"),
    Input("desc-ds-filter",        "value"),
    Input("desc-np-filter",        "value"),
)
def update_pivot_table(solvers, regs, dims, datasets, size):
    dff = _filter_df(solvers, regs, dims, datasets, size)
    summary = (
        dff.groupby(["dim", "solver"])
           .agg(
               mean_runtime=("runtime", "mean"),
               std_runtime=("runtime", "std"),
               mean_error=("error", "mean"),
               std_error=("error", "std"),
           )
           .round(4)
           .reset_index()
    )
    return dash.dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in summary.columns],
        data=summary.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px"},
    )


@dash.callback(
    Output("desc-instability-stats", "children"),
    Input("desc-solver-filter",    "value"),
    Input("desc-reg-filter",       "value"),
    Input("desc-dim-filter",       "value"),
    Input("desc-ds-filter",        "value"),
    Input("desc-np-filter",        "value"),
)
def update_instability_stats(solvers, regs, dims, datasets, size):
    MAX_ITER        = 7000
    ERROR_THRESHOLD = 1e-6
    dff = _filter_df(solvers, regs, dims, datasets, size)
    dff = dff.assign(
        failure = (
            (dff["iterations"] == MAX_ITER) &
            (dff["error"]      >  ERROR_THRESHOLD)
        ).astype(int)
    )
    summary = (
        dff.groupby(["dim", "solver"])
           .agg(
                nan_count      = ("error",      lambda x: x.isna().sum()),
                max_iterations = ("iterations", "max"),
                failure_count  = ("failure",    "sum"),
           )
           .round(4)
           .reset_index()
    )
    return dash.dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in summary.columns],
        data=summary.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px"},
    )

###############################################################
# -------------- RESOURCE USAGE VIOLIN PLOT CALLBACK ---------- 
###############################################################
@dash.callback(
    Output("desc-resource-violins", "figure"),
    Input("desc-solver-filter", "value"),
    Input("desc-reg-filter",    "value"),
    Input("desc-dim-filter",    "value"),
    Input("desc-ds-filter",     "value"),
    Input("desc-np-filter",     "value"),
)
def update_resource_violinplot(solvers, regs, dims, datasets, size):
    # 1) Pull & filter your master DataFrame however you like
    dff = _filter_df(solvers, regs, dims, datasets, size)

    # 2) Prepare memory metrics
    mem = dff.melt(
        id_vars=["dataset", "solver"],
        value_vars=["peak_gpu_mem", "combined_peak_gpu_ram"],
        var_name="metric",
        value_name="value",
    )
    mem["metric_type"] = "Memory (MiB)"

    # 3) Prepare utilization metrics
    util = dff.melt(
        id_vars=["dataset", "solver"],
        value_vars=["peak_util_pct", "mean_util_pct"],
        var_name="metric",
        value_name="value",
    )
    util["metric_type"] = "Utilization (%)"

    # 4) Combine into long form
    long = pd.concat([mem, util], ignore_index=True)

    # 5) Draw the violin plot with facets
    fig = px.violin(
        long,
        x="value",
        y="dataset",
        color="solver",               # color by solver
        facet_col="metric_type",    # two side-by-side subplots
        orientation="h",
        box=True,
        points="all",
        hover_data=["metric"],
        # color_discrete_map=color_map,
        category_orders={"metric_type": ["Memory (MiB)", "Utilization (%)"]},
        template="plotly_white",
    )

    # 6) Tidy up facet titles (remove "metric_type=" prefix)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    # 7) Global layout tweaks
    fig.update_layout(
        title="Resource Usage by Dataset and Solver",
        xaxis_title="",    # per-facet override below
        yaxis_title="Dataset",
        legend_title="Solver",
        height=500,
        width=900,
        margin=dict(t=60, b=50, l=200, r=50),
    )

    # 8) Set each facet’s x-axis title individually
    fig.update_xaxes(title_text="Memory (MiB)",     matches=None, col=1)
    fig.update_xaxes(title_text="Utilization (%)",  matches=None, col=2)

    # 9) Add grid styling
    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgrey",
        gridwidth=1,
        griddash="dash",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="lightgrey",
        gridwidth=1,
        griddash="dash",
        zeroline=False,
    )
    return fig
