import dash
from dash import Output, Input, State
from dash.dependencies import MATCH
from dash import (
    html,
    dcc,
)
import dash_bootstrap_components as dbc

from components import filters
from components.graph_card import (
    graph_card_single,
    graph_card_double,
    download,
    graph_with_export,
)
from dataset import load_all_df, preprocess


df_master = load_all_df()
df_master = preprocess(df_master)

print("df master")
print(f"{df_master['cost_rerr']=}")

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

    # download,
    dcc.Download(id="desc-download-pdf"),

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

            # graph_card_single("Max iterations hit rate per solver",
            #                   "desc-maxiter-heatmap"),

            graph_with_export(
                "Max iterations hit reate per solver",
                "desc-maxiter-heatmap",
                "export-pdf-desc-maxiter-heatmap",
            ),

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

            dbc.Card([
                dbc.CardBody([
                    html.Div(id="desc-size-growth-table-wrapper"),
                ])
            ])
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

            graph_card_single("Error vs. Problem Size", "desc-error-vs-size"),

            graph_card_single("Cost RERR vs. Regularization by Solver", "desc-accuracy-vs-reg"),

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

        # Tab 6: Summary Tables
        dcc.Tab(
            label="Summary Tables",
            value="tab-summary-tables",
            children=[
                # Summary runtime table
                dbc.Card([
                    dbc.CardHeader([
                        "Per-Solver Summary (mean ± std)",
                        html.Button("Copy", id={"type": "copy-btn", "target": "summary-runtime"},
                                    className="btn btn-outline-secondary btn-sm"),
                    ]),
                    dbc.CardBody([
                        html.Div(id="desc-summary-solver-distribution-table"),
                        html.Small(id={"type": "copy-msg", "target": "summary-runtime"},
                                   className="text-success ms-2")
                    ])
                ]),

                # Normalized runtime table
                dbc.Card([
                    dbc.CardHeader([
                        "Normalized Runtime (Dolan–Moré style)",
                        html.Button("Copy", id={"type": "copy-btn", "target": "normalized-runtime"},
                                    className="btn btn-outline-secondary btn-sm"),
                    ]),
                    dbc.CardBody([
                        html.Div(id="desc-normalized-runtime-summary-table"),
                        html.Small(id={"type": "copy-msg", "target": "normalized-runtime"},
                                   className="text-success ms-2")
                    ])
                ]),
            ]
        ),

        dcc.Tab(
            label="Performance Profiles",
            value="tab-performance-profiles",
            children=[
                dbc.Card([
                    dbc.CardHeader([
                        "Overall Performance Profile by Solver"
                    ]),
                    dbc.CardBody(dcc.Graph(id="desc-perfprof-solver", mathjax=True)),
                ]),

                dbc.Card([
                    dbc.CardHeader([
                        "Performance Profile by Solver & Problem Size"
                    ]),
                    dbc.CardBody(dcc.Graph(id="desc-perfprof-solver-size", mathjax=True)),
                ]),

                dbc.Card([
                    dbc.CardHeader([
                        "Performance Profile by Solver & Distribution"
                    ]),
                    dbc.CardBody(dcc.Graph(id="desc-perfprof-solver-dist", mathjax=True)),
                ]),
            ],
        ),

        dcc.Tab(
            label="Epsilon Effect",
            value="tab-epsilon-effect",
            children=[
                graph_card_single(
                    title=r"Runtime vs Regularization $\varepsilon$ ε — Median with IQR band",
                    graph_id="desc-reg-runtime",
                    convergence_switch_id="desc-reg-runtime-only-converged",
                    mathjax=True
                ),

                dbc.Card([
                    dbc.CardBody([
                        html.Div(id="desc-reg-growth-table-wrapper"),
                        # html.Small(id={"type": "copy-msg", "target": "normalized-runtime"},
                        #            className="text-success ms-2")
                    ])
                ])

            ],
        ),

        # dcc.Tab(
        #     label="Dimensionality",
        #     value="tab-dimensionality",
        #     children=[
        #
        #     ],
        # ),
    ])
])


dash.clientside_callback(
    """
    function(n, cols, data) {
        if (!n) { return window.dash_clientside.no_update; }
        if (!cols || !data) { return "Nothing to copy"; }

        const header = cols.map(c => (c.name ?? c.id)).join('\\t');
        const rows = (data || []).map(r =>
            cols.map(c => {
                let v = r[c.id];
                return (v === null || v === undefined) ? "" : String(v).replaceAll('\\n',' ');
            }).join('\\t')
        );
        const tsv = [header].concat(rows).join('\\n');

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(tsv);
            return `Copied ${rows.length} rows`;
        }
        return "Clipboard not available";
    }
    """,
    Output({"type": "copy-msg", "target": MATCH}, "children"),
    Input({"type": "copy-btn", "target": MATCH}, "n_clicks"),
    State({"type": "table", "target": MATCH}, "columns"),
    State({"type": "table", "target": MATCH}, "data"),
)
