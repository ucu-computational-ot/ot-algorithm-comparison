import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
import statsmodels.formula.api as smf

from dataset import load_all_df, preprocess

# Register page
dash.register_page(__name__, path='/inferential', name='Inferential')

# Load & preprocess once
df = load_all_df()
preprocess(df)

# Shared filter controls
filter_controls = html.Div([
    html.Div([
        html.Label("Solver"),
        dcc.Dropdown(id="inf-solver-filter", multi=True, options=[
            # populate from df['solver'].unique()
            {"label": s, "value": s} for s in sorted(df["solver"].unique())
        ], value=sorted(df["solver"].unique())),
    ], style={'width':'20%', 'display':'inline-block'}),
    html.Div([
        html.Label("Regularization"),
        dcc.Dropdown(id="inf-reg-filter", multi=True, options=[
            {"label": r, "value": r} for r in sorted(df["reg"].unique())
        ], value=sorted(df["reg"].unique())),
    ], style={'width':'20%', 'display':'inline-block', 'margin-left':'1%'}),
    html.Div([
        html.Label("Dimension"),
        dcc.Dropdown(id="inf-dim-filter", multi=True, options=[
            {"label": d, "value": d} for d in sorted(df["dim"].unique())
        ], value=sorted(df["dim"].unique())),
    ], style={'width':'20%', 'display':'inline-block', 'margin-left':'1%'}),
    html.Div([
        html.Label("Distribution"),
        dcc.Dropdown(id="inf-ds-filter", multi=True, options=[
            {"label": d, "value": d} for d in sorted(df["distribution"].unique())
        ], value=sorted(df["distribution"].unique())),
    ], style={'width':'20%', 'display':'inline-block', 'margin-left':'1%'}),
    html.Div([
        html.Label("Points"),
        dcc.Dropdown(id="inf-np-filter", multi=True, options=[
            {"label": n, "value": n} for n in sorted(df["size"].unique())
        ], value=sorted(df["size"].unique())),
    ], style={'width':'15%', 'display':'inline-block', 'margin-left':'1%'}),
], style={'margin-bottom':'30px'})


# Helper: filter master df
def _filter_df(solvers, regs, dims, datasets, size):
    dff = df[
        df["solver"].isin(solvers) &
        df["reg"].isin(regs) &
        df["dim"].isin(dims) &
        df["distribution"].isin(datasets) &
        df["size"].isin(size)
    ]
    return dff


# Card factory
def make_card(title, button_id, output_id):
    return html.Div([
        html.H4(title),
        html.Button(f"Run {title}", id=button_id, n_clicks=0),
        html.Div(id=output_id, style={'padding':'10px', 'border':'1px solid #ddd'})
    ], style={'margin-bottom':'20px', 'padding':'10px', 'box-shadow':'2px 2px 5px #ccc'})

layout = html.Div([
    html.H2("Inferential Analysis"),
    filter_controls,
    make_card("Normality Test (Shapiro–Wilk)",    "btn-normality",      "out-normality"),
    make_card("Homoscedasticity Test (Levene)",   "btn-homoscedastic",  "out-homoscedastic"),
    make_card("Pairwise Tests",                   "btn-pairwise",       "out-pairwise"),
    make_card("Multi-Method Tests",               "btn-multimethod",    "out-multimethod"),
    make_card("Correlation Test",                 "btn-correlation",    "out-correlation"),
])

# Utility: pivot wide for all selected solvers
def pivot_wide(dff, col):
    idx = ['dataset','dim','reg','size']
    wide = dff.pivot_table(index=idx, columns='solver', values=col)
    return wide.dropna(axis=0, subset=dff['solver'].unique())

# 1) Normality callback
@dash.callback(
    Output("out-normality", "children"),
    Input("btn-normality", "n_clicks"),
    State("inf-solver-filter", "value"),
    State("inf-reg-filter",    "value"),
    State("inf-dim-filter",    "value"),
    State("inf-ds-filter",     "value"),
    State("inf-np-filter",     "value"),
)
def run_normality(nc, solvers, regs, dims, dsets, nps):
    if nc==0:
        return "Click to run Shapiro–Wilk on runtime differences (requires exactly 2 solvers)."
    dff = _filter_df(solvers, regs, dims, dsets, nps)
    if len(solvers)!=2:
        return "Normality test needs exactly 2 solvers selected."
    rt_w = pivot_wide(dff, "runtime")
    diff = rt_w[solvers[0]] - rt_w[solvers[1]]
    W, p = stats.shapiro(diff)
    return html.Div([
        html.P(f"W statistic: {W:.4f}"),
        html.P(f"p-value: {p:.4e}"),
        html.P("⇒ Normality " + ("holds" if p>0.05 else "rejected"))
    ])

# 2) Homoscedasticity callback
@dash.callback(
    Output("out-homoscedastic", "children"),
    Input("btn-homoscedastic", "n_clicks"),
    State("inf-solver-filter", "value"),
    State("inf-reg-filter",    "value"),
    State("inf-dim-filter",    "value"),
    State("inf-ds-filter",     "value"),
    State("inf-np-filter",     "value"),
)
def run_homoscedastic(nc, solvers, regs, dims, dsets, nps):
    if nc==0:
        return "Click to run Levene’s test across solver groups."
    dff = _filter_df(solvers, regs, dims, dsets, nps)
    rt_w = pivot_wide(dff, "runtime")
    groups = [rt_w[s] for s in solvers]
    stat, p = stats.levene(*groups)
    return html.Div([
        html.P(f"Levene’s W: {stat:.4f}"),
        html.P(f"p-value: {p:.4e}"),
        html.P("⇒ Equal variances " + ("assumed" if p>0.05 else "rejected"))
    ])

# 3) Pairwise tests callback
@dash.callback(
    Output("out-pairwise", "children"),
    Input("btn-pairwise", "n_clicks"),
    State("inf-solver-filter", "value"),
    State("inf-reg-filter",    "value"),
    State("inf-dim-filter",    "value"),
    State("inf-ds-filter",     "value"),
    State("inf-np-filter",     "value"),
)
def run_pairwise(nc, solvers, regs, dims, dsets, nps):
    if nc==0:
        return "Click to run paired t-test or Wilcoxon signed‐rank (2 solvers only)."
    dff = _filter_df(solvers, regs, dims, dsets, nps)
    if len(solvers)!=2:
        return "Pairwise tests require exactly 2 solvers."
    rt_w = pivot_wide(dff, "runtime")
    diff = rt_w[solvers[0]] - rt_w[solvers[1]]
    # check normality
    _, p_sw = stats.shapiro(diff)
    if p_sw>0.05:
        t, p = stats.ttest_rel(rt_w[solvers[0]], rt_w[solvers[1]])
        name = "Paired t-test"
        stats_txt = f"t = {t:.4f}, p = {p:.4e}"
    else:
        w, p = stats.wilcoxon(rt_w[solvers[0]], rt_w[solvers[1]])
        name = "Wilcoxon signed‐rank"
        stats_txt = f"V = {w:.4f}, p = {p:.4e}"
    return html.Div([
        html.P(f"Normality p = {p_sw:.4e}"),
        html.P(f"{name}: {stats_txt}")
    ])

# 4) Multi-method tests callback
@dash.callback(
    Output("out-multimethod", "children"),
    Input("btn-multimethod", "n_clicks"),
    State("inf-solver-filter", "value"),
    State("inf-reg-filter",    "value"),
    State("inf-dim-filter",    "value"),
    State("inf-ds-filter",     "value"),
    State("inf-np-filter",     "value"),
)
def run_multimethod(nc, solvers, regs, dims, dsets, nps):
    if nc==0:
        return "Click to run Friedman’s test, Nemenyi post hoc, and RM‐ANOVA."
    dff = _filter_df(solvers, regs, dims, dsets, nps)
    rt_w = pivot_wide(dff, "runtime")
    # Friedman
    args = [rt_w[s] for s in solvers]
    chi2, p_f = stats.friedmanchisquare(*args)
    # Nemenyi
    nemenyi = sp.posthoc_nemenyi_friedman(rt_w[solvers].values)
    # RM‐ANOVA
    long = rt_w.reset_index().melt(id_vars=['dataset','dim','reg','size'],
                                   value_vars=solvers,
                                   var_name='solver', value_name='runtime')
    aov = smf.ols("runtime ~ C(solver)+C(dataset)", data=long).fit()
    tbl = smf.stats.anova_lm(aov, typ=2)
    return html.Div([
        html.H5(f"Friedman’s χ² = {chi2:.4f}, p = {p_f:.4e}"),
        html.P("Nemenyi post-hoc p-values:"),
        dcc.Graph(figure={
            'data':[{
                'type':'heatmap',
                'z': nemenyi.values,
                'x': solvers, 'y': solvers,
                'colorscale':'Blues'
            }],
            'layout':{'height':300, 'margin':{'l':50,'r':50,'t':30,'b':30}}
        }),
        html.H5("Repeated-Measures ANOVA"),
        dcc.Markdown(tbl.to_markdown())
    ])

# 5) Correlation tests callback
@dash.callback(
    Output("out-correlation", "children"),
    Input("btn-correlation", "n_clicks"),
    State("inf-solver-filter", "value"),
    State("inf-reg-filter",    "value"),
    State("inf-dim-filter",    "value"),
    State("inf-ds-filter",     "value"),
    State("inf-np-filter",     "value"),
)
def run_correlation(nc, solvers, regs, dims, dsets, nps):
    if nc==0:
        return "Click to run Pearson & Spearman correlations between runtime & error."
    dff = _filter_df(solvers, regs, dims, dsets, nps)
    r1, p1 = stats.pearsonr(dff['runtime'], dff['error'])
    r2, p2 = stats.spearmanr(dff['runtime'], dff['error'])
    return html.Ul([
        html.Li(f"Pearson’s r = {r1:.4f}, p = {p1:.4e}"),
        html.Li(f"Spearman’s ρ = {r2:.4f}, p = {p2:.4e}")
    ])
