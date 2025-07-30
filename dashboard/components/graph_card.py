from dash import html, dcc
import dash_bootstrap_components as dbc

from components import filters


def graph_card_single(
    title: str,
    graph_id: str,
    convergence_switch_id: str | None = None,
    **kwargs
):
    return html.Div([
        html.H4(title),
        filters.only_converged_filter(convergence_switch_id) if convergence_switch_id else None,
        html.Div(dcc.Graph(id=graph_id, **kwargs), style={
            "width": "100%",
            "overflowX": "scroll",
        }),
    ])


def graph_card_double(title: str, graph_ids: tuple[str, str], **kwargs):
    return html.Div([
        html.H4(title),
        dbc.Row([
            dbc.Col(dcc.Graph(id=graph_ids[0], **kwargs), width=6),
            dbc.Col(dcc.Graph(id=graph_ids[1], **kwargs), width=6),
        ], style={
                "width": "100%",
                "overflowX": "scroll",
                })
    ])
