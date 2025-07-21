import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Initialize the Dash app with pages
app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    meta_tags=[{"viewport": "width=device-width, initial-scale=1.0"}],
    external_stylesheets=[
        dbc.themes.FLATLY
    ]
)
server = app.server

# Navbar
navbar = dbc.NavbarSimple(
    brand="OT Dashboard",
    brand_href="/",
    color="primary",
    dark=True,
    children=[
        dbc.NavItem(dbc.NavLink("Descriptive Analysis", href="/descriptive")),
        dbc.NavItem(dbc.NavLink("Inferential Analysis", href="/inferential")),
    ],
    className="mb-4",
)

app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        navbar,
        dash.page_container
    ],
    fluid=True,
    style={"padding": "0 2rem"}
)

if __name__ == '__main__':
    app.run(debug=True, port=8050)
