import dash
from dash import Output
import plotly.express as px
import pandas as pd
from utils.filters import FILTER_INPUTS
from utils.data import get_filtered, SOLVERS_COLOR_MAP


@dash.callback(
    Output("desc-resources-violins", "figure"),
    *FILTER_INPUTS,
)
def update_resource_violinplot(solvers, regs, dims, datasets, size):
    dff = get_filtered(solvers, regs, dims, datasets, size)

    # drop rows where ALL metrics are zero, as before
    metrics_to_check = [
        "peak_gpu_mem", "combined_peak_gpu_mem",
        "peak_ram_mem", "combined_peak_ram_mem",
        "peak_util_pct", "mean_util_pct",
        "peak_cpu_util_pct", "mean_cpu_util_pct"
    ]
    dff = dff[~(dff[metrics_to_check] == 0).all(axis=1)]

    # melt each group
    def melt_group(df, vars, metric_type):
        return (
            df.melt(
                id_vars=["dataset", "solver"],
                value_vars=vars,
                var_name="metric",
                value_name="value",
            )
            .assign(metric_type=metric_type)
            .query("value > 0")
        )

    mem_gpu = melt_group(dff, ["peak_gpu_mem", "combined_peak_gpu_mem"], "GPU Memory (MiB)")
    mem_cpu = melt_group(dff, ["peak_ram_mem",   "combined_peak_ram_mem"], "CPU Memory (MiB)")
    util_gpu = melt_group(dff, ["peak_util_pct",  "mean_util_pct"],        "GPU Utilization (%)")
    util_cpu = melt_group(dff, ["peak_cpu_util_pct","mean_cpu_util_pct"],   "CPU Utilization (%)")

    long = pd.concat([mem_gpu, mem_cpu, util_gpu, util_cpu], ignore_index=True)

    if long.empty:
        return {
            "data": [], 
            "layout": {"title": "No resource data available for these filters"}
        }

    # 2×2 facet: rows=Memory/Util, cols=GPU/CPU
    fig = px.violin(
        long,
        x="value",
        y="dataset",
        color="solver",
        color_discrete_map=SOLVERS_COLOR_MAP,
        facet_row="metric_type",       # this will produce 4 rows if you omit facet_col
        orientation="h",
        box=True,
        points="all",
        hover_data=["metric"],
        category_orders={
          "metric_type": [
              "GPU Memory (MiB)", "CPU Memory (MiB)",
              "GPU Utilization (%)", "CPU Utilization (%)"
          ]
        },
        template="plotly_white",
    )

    # Tidy up the facet labels
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    fig.update_layout(
        title="Resource Usage by Dataset and Solver",
        legend_title="Solver",
        height=800,
        margin=dict(t=80, b=50, l=200, r=50),
    )

    # Individual axis titles
    fig.update_xaxes(matches=None, title_text="MiB", row=1, col=1)
    fig.update_xaxes(matches=None, title_text="MiB", row=2, col=1)
    fig.update_xaxes(matches=None, title_text="Percent", row=3, col=1)
    fig.update_xaxes(matches=None, title_text="Percent", row=4, col=1)

    # grid styling
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1, griddash="dash", zeroline=False)

    return fig
