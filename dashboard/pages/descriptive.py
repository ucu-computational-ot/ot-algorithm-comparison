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

    # Filters (unchanged)
    dbc.Card(dbc.CardBody(
        dbc.Row([
            filters.solver_filter(solvers, solvers),
            filters.regularisation_filter(regs, []),
            filters.dimension_filter(dims, []),
            filters.size_filter(npoints, npoints),
            filters.dataset_filter(datasets, datasets),
        ], class_name="g-3"), style={"background": "#f0f0f0"},
    ), class_name="mb-4"),

    # ---------- TABS ----------
    dcc.Tabs(id="desc-tabs", value="tab-instability", children=[

        # Tab 1: Instability
        dcc.Tab(label="Instability", value="tab-instability", children=[
            # dbc.Card([
            #     dbc.CardHeader("Instability Statistics"),
            #     dbc.CardBody(
            #         dbc.Table(
            #             id="desc-instability-stats",
            #             striped=True, bordered=True, hover=True, responsive=True,
            #         )
            #     ),
            # ], class_name="mb-4"),

            graph_card_single("Instability Heatmap", "desc-instability-heatmap"),
            html.Hr(),

            graph_card_single("Max iterations hit rate per solver",
                              "desc-maxiter-heatmap"),

            graph_card_single(
                "Instability (NaN rate) per regularization level (per solver)",
                "desc-instability-per-epsilon-heatmap"
            ),

            graph_card_single(
                "Max iterations hit rate per regularization level (per solver)",
                "desc-maxiter-per-epsilon-heatmap"
            ),
        ]),

        # Tab 2: Runtime
        dcc.Tab(label="Runtime", value="tab-runtime", children=[
            graph_card_single("Runtime Violin Plot", "desc-runtime-violinplot"),
            # dcc.Graph(id="desc-runtime-violinplot"),
            html.Hr(),
            graph_card_double("Runtime Distribution", ("desc-runtime-hist", "desc-runtime-box")),
            html.Hr(),
        ]),

        # Tab 3: Iterations & Scaling
        dcc.Tab(label="Iterations & Scaling", value="tab-iter-scale", children=[
            graph_card_single("Iteration vs Problem Size", "desc-iteration-vs-problem-size"),
            html.Hr(),
            graph_card_single(
                title="Scaling: Runtime vs Size",
                graph_id="desc-scaling-scatter",
                convergence_switch_id="desc-scaling-scatter-only-converged",
            ),
            html.Hr(),
        ]),

        # Tab 4: Error
        dcc.Tab(label="Error", value="tab-error", children=[
            graph_card_single(title="Error vs Runtime", graph_id="desc-error-runtime-scatter"),

            graph_card_single(
                title="Pareto scatter: runtime vs cost error (lower-left is better)",
                graph_id="desc-pareto-scatter"
            ),

            graph_card_single(
                title="Distribution Family Sensitivity (boxplots by family, colored by solver)",
                graph_id="desc-dist-sensitivity",
            ),

            html.Hr(),
            graph_card_single("Accuracy vs. Size (cost error vs LP baseline)", "desc-accuracy-vs-size"),
            # html.H4("Cost Error"),
            # dcc.Graph(id="desc-cost-error-vs-runtime"),
            graph_card_single("Cost RERR vs. Problem Size by Solver (Box-Plot)", "desc-cost-rerr-vs-problem-size"),
            # dcc.Graph(id="desc-cost-rerr-vs-problem-size"),
            html.Hr(),
            # dcc.Graph(id='desc-cost-err-problem-size-violinplot'),
        ]),

        # Tab 5: Pivot & Resources
        dcc.Tab(label="Pivot & Resources", value="tab-pivot-resources", children=[
            html.H4("Pivot Table"),
            filters.only_converged_filter("desc-median-pivot-only-converged"),
            html.Div(id="desc-pivot-table", style={"overflowX": "auto"}),
            html.Hr(),
            html.H4("Resource Usage"),
            dcc.Graph(id="desc-resources-violins"),  # uncomment when callback ready
        ]),
    ])
])
