from dataset import load_all_df, preprocess, filter_converged
import plotly.express as px

df_master = load_all_df()
df_master = preprocess(df_master)


def get_filtered(solvers, regs, dims, distributions, sizes, converged=False):
    dff = df_master[
        df_master["solver"].isin(solvers) &
        df_master["reg"].isin(regs) &
        df_master["dim"].isin(dims) &
        df_master["distribution"].isin(distributions) &
        df_master["size"].isin(sizes)
    ]
    return filter_converged(dff) if converged else dff


# palette = px.colors.qualitative.G10
# palette = px.colors.qualitative.Plotly
palette = px.colors.qualitative.Vivid
# palette = px.colors.qualitative.Set1
SOLVERS_COLOR_MAP = {
    solver: palette[i % len(palette)]
    for i, solver in enumerate(sorted(df_master["solver"].unique()))
}
