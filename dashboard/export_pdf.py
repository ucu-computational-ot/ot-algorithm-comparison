import dash
from dash import Input, Output, State, dcc, ctx
from dash.dependencies import MATCH
import plotly.graph_objects as go
import plotly.io as pio


@dash.callback(
    Output("desc-download-pdf", "data"),
    Input("export-pdf-desc-maxiter-heatmap", "n_clicks"),
    State("desc-maxiter-heatmap", "figure"),
    prevent_initial_call=True,
)
def export_maxiter_pdf(n, fig_dict):
    if not n or not fig_dict:
        return dash.no_update
    fig = go.Figure(fig_dict)
    w = fig.layout.width or 1200
    h = fig.layout.height or 400
    fig.update_layout(
        autosize=False,
        width=w,
        height=h,
        font=dict(family="Times New Roman", size=14, color="black"),
    )
    pdf = pio.to_image(fig, format="pdf", scale=2)
    return dcc.send_bytes(lambda b: b.write(pdf), filename="maxiter-heatmap.pdf")
