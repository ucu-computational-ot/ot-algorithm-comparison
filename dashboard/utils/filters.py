from dash.dependencies import Input

FILTER_INPUTS = [
    Input("desc-solver-filter", "value"),
    Input("desc-reg-filter",    "value"),
    Input("desc-dim-filter",    "value"),
    Input("desc-ds-filter",     "value"),
    Input("desc-np-filter",     "value"),
]
