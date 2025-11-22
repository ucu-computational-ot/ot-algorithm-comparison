import dash
from dash import html
import dash_bootstrap_components as dbc
import pandas as pd

from dataset import load_all_df, preprocess

dash.register_page(__name__, path="/", name="Home", title="Home")

# --- Data prep ----------------------------------
df = load_all_df()
df = preprocess(df)

# Build a coverage table: for each distribution, list available solvers and regs
coverage_df = (
    df
    .groupby("distribution")
    .agg(
        solvers=("solver", lambda x: ", ".join(sorted(x.unique()))),
        regs=("reg", lambda x: ", ".join(sorted(map(str, x.unique()))))
    )
    .reset_index()
    .rename(columns={"distribution": "Dataset", "solvers": "Solvers", "regs": "Regularizations"})
)
coverage_table = dbc.Table.from_dataframe(
    coverage_df,
    striped=True,
    bordered=True,
    hover=True,
    responsive=True,
    class_name="mt-4"
)

# -- Hero Section ------------------------------------------------
hero = html.Div(
    [
        html.H1("Optimal Transport Dashboard", className="display-4"),
        html.P(
            "Explore performance and statistical comparisons of OT solvers.",
            className="lead",
        ),
        dbc.Button("Descriptive Analysis", href="/descriptive", color="primary", className="me-2"),
        dbc.Button("Inferential Analysis", href="/inferential", color="secondary"),
    ],
    className="p-5 mb-4 bg-light rounded-3",
)

# -- Summary Cards ------------------------------------------------
stats = [
    ("Datasets", len(df['dataset'].unique())),
    ("Solvers",  len(df['solver'].unique())),
    ("Total Runs", len(df)),
]

cards = [
    dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(title),
                dbc.CardBody(html.H2(value, className="card-title")),
            ],
            className="text-center",
        ),
        width=4,
    )
    for title, value in stats
]

# -- Layout -------------------------------------------------------
layout = dbc.Container(
    [
        hero,
        dbc.Row(cards, className="g-4"),
        # Coverage Table Section
        html.H2("Dataset Coverage", className="mt-5"),
        html.P("Which solvers & regularizations are available per dataset:"),
        coverage_table,
    ],
    fluid=True,
)
