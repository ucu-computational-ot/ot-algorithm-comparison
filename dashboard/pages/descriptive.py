import dash
from dash import (
    html,
    dcc,
)
import dash_bootstrap_components as dbc

from components import filters
from components.graph_card import graph_card_single, graph_card_double
from dataset import load_all_df, preprocess


df_master = load_all_df()
preprocess(df_master)

# sorted lists
solvers = sorted(df_master["solver"].unique())
regs = sorted(df_master["reg"].dropna().astype(float).unique())
dims = sorted(df_master["dim"].dropna().astype(int).unique())
datasets = sorted(df_master["distribution"].unique())
npoints = sorted(df_master["size"].dropna().astype(int).unique())

# --- layout --------------------------------------------------------
dash.register_page(__name__, path="/descriptive", name="Descriptive")

layout = dbc.Container(fluid=True, class_name="p-4", children=[

    html.H2("Descriptive Statistics", className="mb-4"),

    # Filters
    dbc.Card(dbc.CardBody(
        dbc.Row([
            filters.solver_filter(solvers, solvers),
            filters.regularisation_filter(regs, []),
            filters.dimension_filter(dims, []),
            filters.size_filter(npoints, npoints),
            filters.dataset_filter(datasets, datasets),
        ], class_name="g-3"), style={"background": "#f0f0f0"},
    ), class_name="mb-4"),

    # --- Section: Instability Statistics ---------------------------
    dbc.Card([
        dbc.CardHeader("Instability Statistics"),
        dbc.CardBody(
            dbc.Table(
                id="desc-instability-stats",
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
            )
        ),
    ], className='mb-4'),

    # --- Section: Runtime Violin Plot ------------------------------
    graph_card_single("Runtime Violin Plot", "desc-runtime-violinplot"),
    html.Hr(),

    # --- Section: Iteration vs Problem Size ------------------------
    graph_card_single("Iteration vs Problem Size", "desc-iteration-vs-problem-size"),
    html.Hr(),

    # --- Section: Runtime Distribution -----------------------------
    graph_card_double("Runtime Distribution", (
        "desc-runtime-hist",
        "desc-runtime-box"
    )),
    html.Hr(),

    # --- Section: Scaling & Error vs Runtime -----------------------
    # html.H4("Scaling: Runtime vs Size"),
    # filters.only_converged_filter("desc-scaling-scatter-only-converged"),
    # dcc.Graph(id='desc-scaling-scatter'),
    graph_card_single(
        title="Scaling: Runtime vs Size",
        graph_id="desc-scaling-scatter",
        convergence_switch_id="desc-scaling-scatter-only-converged",
    ),

    # html.H4("Error vs Runtime"),
    # dcc.Graph(id='desc-error-runtime-scatter'),
    graph_card_single(
        title="Error vs Runtime",
        graph_id="desc-error-runtime-scatter",
    ),
    html.Hr(),

    # --- Section: Pivot Table ---------------------
    html.H4("Pivot Table"),
    filters.only_converged_filter("desc-median-pivot-only-converged"),
    html.Div(id="desc-pivot-table", style={"overflowX": "auto"}),
    html.Hr(),

    # --- Section: Resource Usage -----------------------------------
    html.H4("Resource Usage"),
    dcc.Graph(id='desc-resources-violins'),


    # --- Section: Cost Relative Error ------------------------------
    html.H4("Cost Error"),
    dcc.Graph(id='desc-cost-error-vs-runtime'),

    dcc.Graph(id='desc-cost-err-problem-size-violinplot'),

    dcc.Graph(id='desc-cost-rerr-vs-problem-size'),
])


