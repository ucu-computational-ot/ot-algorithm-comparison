import numpy as np
import pandas as pd

def get_agg_table(df):
        df = df.groupby('name').agg(
                time_mean = ('time', 'mean'),
                time_std = ('time', lambda x: np.std(x, ddof=1)),
                cost_rerr_mean = ('cost_rerr', "mean"),
                cost_rerr_std = ('cost_rerr', lambda x: np.std(x, ddof=1)),
                coupling_avg_err_mean = ('coupling_avg_err', 'mean'),
                coupling_avg_err_std = ('coupling_avg_err', lambda x: np.std(x, ddof=1)),
                memory_mean = ('memory', 'mean'),
                memory_std = ('memory', lambda x: np.std(x, ddof=1))
        ).reset_index()
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


        

        
