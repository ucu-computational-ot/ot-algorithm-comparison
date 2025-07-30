import os
import dash
from dash import html, dcc, clientside_callback, Input, Output
import dash_bootstrap_components as dbc

# Initialize the Dash app with pages
app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    meta_tags=[{"viewport": "width=device-width, initial-scale=1.0"}],
    external_stylesheets=[
        # dbc.themes.JOURNAL,
        dbc.themes.SPACELAB,
        # dbc.themes.SIMPLEX,
        # dbc.themes.LITERA,
        # dbc.themes.FLATLY,
        dbc.icons.FONT_AWESOME,
    ]
)
server = app.server

# color_mode_switch = html.Span(
#     [
#         dbc.Label(className="fa fa-moon", html_for="switch"),
#         dbc.Switch(
#             id="switch", value=True, className="d-inline-block ms-1",
#             persistence=True
#         ),
#         dbc.Label(className="fa fa-sun", html_for="switch"),
#     ]
# )

# clientside_callback(
#     """
#     (switchOn) => {
#        document.documentElement.setAttribute("data-bs-theme", switchOn ? "light" : "dark");
#        return window.dash_clientside.no_update
#     }
#     """,
#     Output("switch", "id"),
#     Input("switch", "value"),
# )

# Navbar
navbar = dbc.NavbarSimple(
    brand="OT Dashboard",
    brand_href="/",
    color="primary",
    dark=True,
    children=[
        # color_mode_switch,
        dbc.NavItem(dbc.NavLink("Descriptive Analysis", href="/descriptive")),
        dbc.NavItem(dbc.NavLink("Inferential Analysis", href="/inferential")),
    ],
    className="mb-2",
)

app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        navbar,
        dash.page_container
    ],
    fluid=True,
    style={"padding": "0"}
)

import callbacks

if __name__ == '__main__':
    debug = os.environ.get('DEBUG', False)
    app.run(debug=debug, port=8050)
