import dash_bootstrap_components as dbc
from dash import html, dcc


def solver_filter(solvers, default):
    return dbc.Col([
        html.Label("Solvers"),
        dbc.Checklist(
            id="desc-solver-filter",
            options=[{"label": s, "value": s} for s in solvers],
            value=default,
            switch=True, persistence=True,
            # style={"maxHeight": "180px", "overflowY": "auto"}
        )
    ], width=3)


def regularisation_filter(regs, default):
    return dbc.Col([
        html.Label("Regularization ε"),
        dbc.Checklist(
            id="desc-reg-filter",
            options=[{"label": r, "value": r} for r in regs],
            value=default,
            switch=True,
            persistence=True,
            # style={"maxHeight": "180px", "overflowY": "auto"}
        ),
    ], width=2)


def dimension_filter(dims, default):
    return dbc.Col([
        html.Label("Dimension"),
        dbc.Checklist(
            id="desc-dim-filter",
            options=[{"label": f"{d}D", "value": d} for d in dims],
            value=default,
            switch=True,
            persistence=True
        ),
    ], width=2)


def size_filter(sizes, default):
    return dbc.Col([
        html.Label("Size (# points)"),
        dbc.Checklist(
            id="desc-np-filter",
            options=[{"label": n, "value": n} for n in sizes],
            value=default,
            switch=True,
            persistence=True
        ),
    ], width=2)


def dataset_filter(datasets, default):
    return dbc.Col([
        html.Label("Datasets"),
        dcc.Dropdown(
            id="desc-ds-filter",
            options=[{"label": ds, "value": ds} for ds in datasets],
            value=default,
            multi=True,
            placeholder="Select datasets...",
            persistence=True,
        ),
    ], width=3)


def only_converged_filter(id):
    return dbc.Col([
    # return html.Span([
        html.Label("Only Converged"),
        dbc.Checklist(
            id=id,
            options=[{"label": "", "value": True}],
            value=[],
            switch=True,
            persistence=True,
            inline=True,
        ),
    ],)
