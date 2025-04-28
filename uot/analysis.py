import numpy as np
import pandas as pd
from IPython.display import display, HTML

def get_agg_table(df, metrics: list[str]):
        aggregation_rules = {f"{metric}_mean": (metric, 'mean') for metric in metrics} | \
                            {f"{metric}_std": (metric, lambda x: np.std(x, ddof=1)) for metric in metrics }

        
        df = df.groupby('name').agg(**aggregation_rules).reset_index()
        return df.sort_values('name')


def get_comparison_table(dfs: dict[str, pd.DataFrame], field: str):
        comparison_df = pd.DataFrame()
        comparison_df['dataset'] = list(dfs.values())[0].name

        for name, df in dfs.items():
                mean = df[f"{field}_mean"].astype(str)
                std = df[f"{field}_std"].astype(str)
                comparison_df[name] = mean + "±" + std
        return comparison_df
        
def get_mean_comparison_table(dfs: dict[str, pd.DataFrame], field: str):
        comparison_df = pd.DataFrame()
        comparison_df['dataset'] = list(dfs.values())[0].name

        for name, df in dfs.items():
                comparison_df[name] = df[f"{field}_mean"]
        return comparison_df

def get_std_comparison_table(dfs: dict[str, pd.DataFrame], field: str):
        comparison_df = pd.DataFrame()
        comparison_df['dataset'] = list(dfs.values())[0].name

        for name, df in dfs.items():
                comparison_df[name] = df[f"{field}_std"]
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