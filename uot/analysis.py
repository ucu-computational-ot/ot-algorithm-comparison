import numpy as np
import pandas as pd
from IPython.display import display, HTML

def get_agg_table(df, metrics: list[str]):
        aggregation_rules = {f"{metric}_mean": (metric, 'mean') for metric in metrics} | \
                            {f"{metric}_std": (metric, lambda x: np.std(x, ddof=1)) for metric in metrics }

        
        df = df.groupby('dataset').agg(**aggregation_rules).reset_index()
        return df


def get_mean_comparison_table(df: dict[str, pd.DataFrame], field: str):
        comparison_df = pd.DataFrame()
        comparison_df['dataset'] = df.dataset.unique()

        dfs = [df[df.name == name] for name in df.name.unique()]
        agg_dfs = {df.name.iloc[0]: get_agg_table(df, [field]) for df in dfs}

        for name, df in agg_dfs.items():
                mean_column = f"{field}_mean"
                df = df[['dataset', mean_column]].rename(columns={mean_column: name})
                comparison_df = pd.merge(comparison_df, df, on='dataset', how='inner')
        return comparison_df

def get_std_comparison_table(df: pd.DataFrame, field: str):
        comparison_df = pd.DataFrame()
        comparison_df['dataset'] = df.dataset.unique()

        dfs = [df[df.name == name] for name in df.name.unique()]
        agg_dfs = {df.name.iloc[0]: get_agg_table(df, [field]) for df in dfs}

        for name, df in agg_dfs.items():
                std_column = f"{field}_std"
                df = df[['dataset', std_column]].rename(columns={std_column: name})
                comparison_df = pd.merge(comparison_df, df, on='dataset', how='inner')
        return comparison_df

def display_mean_and_std(agg_dfs, column: str):
        def highlight_top_two(s):
                is_first = s == s.min()
                is_second = s == s[s != s.min()].min()
                return [
                        'background-color: lightgreen' if f else
                        'background-color: lightblue' if s_ else
                        ''
                        for f, s_ in zip(is_first, is_second)
                ]
        mean_comparison = get_mean_comparison_table(agg_dfs, column)
        std_comparison = get_std_comparison_table(agg_dfs, column)

        mean_html = mean_comparison.style.apply(highlight_top_two, axis=1, subset=mean_comparison.columns[1:]).to_html()

        std_html = std_comparison.style.apply(highlight_top_two, axis=1, subset=std_comparison.columns[1:]).to_html()

        combined_html = f"""
        <h3 style="text-align:center;">{column} mean and std</h3>
        <div style="display: flex; justify-content: space-around;">
        <div>{mean_html}</div>
        <div>{std_html}</div>
        </div>
        """
        return display(HTML(combined_html))

def display_all_metrics(results: dict[str, pd.DataFrame], metrics_names: list[str]):
        agg_dfs = {solver_name: get_agg_table(result.df, metrics=metrics_names) for solver_name, result in results.items()}

        for metrics_name in metrics_names:
                display_mean_and_std(agg_dfs, metrics_name)