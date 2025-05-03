import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from uot.analysis import get_agg_table, get_mean_comparison_table, get_std_comparison_table, display_mean_and_std, display_all_metrics

plt.style.use('ggplot')
sns.set_palette("colorblind")

results_df = pd.read_csv('result_2025-05-01_02-52-55.csv')

print(f"Shape of results: {results_df.shape}")
print(f"Algorithms: {results_df['name'].unique()}")
print(f"Datasets: {results_df['dataset'].unique()}")

results_df.head()

metrics_to_analyze = ['time', 'cost_rerr', 'coupling_avg_err']

one_dim_datasets = [ds for ds in results_df['dataset'].unique() if '1D' in ds]
two_dim_datasets = [ds for ds in results_df['dataset'].unique() if '2D' in ds or 'x32' in ds]
mesh_datasets = [ds for ds in results_df['dataset'].unique() if 'Mesh' in ds]

dataset_groups = {
    '1D Datasets': one_dim_datasets,
    '2D Datasets': two_dim_datasets,
    '3D Mesh Datasets': mesh_datasets
}

algorithms = results_df['name'].unique()

import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_runtime_distributions(datasets, results_df, algorithms, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    for dataset in datasets:
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        ax1 = axes[0]
        ax1.set_yscale('log')
        for algo in algorithms:
            algo_data = dataset_data[dataset_data['name'] == algo]['time']
            if len(algo_data) > 0:
                sns.kdeplot(algo_data, ax=ax1, label=algo, fill=True, alpha=0.3)
        
        ax1.set_title(f'KDE of Runtime for {dataset}')
        ax1.set_xlabel('Runtime (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        ax2 = axes[1]
        plot_data = []
        labels = []

        for algo in algorithms:
            algo_data = dataset_data[dataset_data['name'] == algo]['time']
            if len(algo_data) > 0:
                plot_data.append(algo_data)
                labels.append(algo)
            else:
                print(f"Algorithm: {algo}, Dataset: {dataset}, Count: {len(algo_data)} - No data")

        ax2.boxplot(plot_data, tick_labels=labels, showmeans=True)
        ax2.set_title(f'Box Plot of Runtime for {dataset}')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.set_yscale('log')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        save_file = os.path.join(save_path, f"{dataset}_runtime_distributions.png")
        plt.savefig(save_file)
        plt.close(fig)


def plot_cost_err_distributions(datasets, results_df, algorithms, save_path):
    os.makedirs(save_path, exist_ok=True)

    for dataset in datasets:
        dataset_data = results_df[results_df['dataset'] == dataset]

        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        ax1 = axes[0]
        ax1.set_yscale('log')
        for algo in algorithms:
            algo_data = dataset_data[dataset_data['name'] == algo]['cost_rerr']
            if len(algo_data) > 1 and algo_data.nunique() > 1:
                sns.kdeplot(algo_data, ax=ax1, label=algo, fill=True, alpha=0.3)
            elif algo_data.nunique() == 1:
                print(f"Algorithm: {algo}, Dataset: {dataset}, Count: {len(algo_data)} - Constant value")
                const_val = algo_data.iloc[0]
                ax1.axvline(const_val, label=f"{algo} (constant)", linestyle='--')

        ax1.set_title(f'KDE of Cost Error for {dataset}')
        ax1.set_xlabel('Cost Error')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        ax2 = axes[1]
        plot_data = []
        labels = []

        for algo in algorithms:
            algo_data = dataset_data[dataset_data['name'] == algo]['cost_rerr']
            if len(algo_data) > 0:
                plot_data.append(algo_data)
                labels.append(algo)

        ax2.boxplot(plot_data, tick_labels=labels, showmeans=True)
        ax2.set_title(f'Box Plot of Cost Error for {dataset}')
        ax2.set_ylabel('Cost Error')
        ax2.set_yscale('log')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{dataset}_cost_err_distributions.png"))
        plt.close(fig)


def plot_coupling_err_distributions(datasets, results_df, algorithms, save_path):
    os.makedirs(save_path, exist_ok=True)

    for dataset in datasets:
        dataset_data = results_df[results_df['dataset'] == dataset]

        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        ax1 = axes[0]
        ax1.set_yscale('log')
        for algo in algorithms:
            algo_data = dataset_data[dataset_data['name'] == algo]['coupling_avg_err']
            if len(algo_data) > 0:
                sns.kdeplot(algo_data, ax=ax1, label=algo, fill=True, alpha=0.3)

        ax1.set_title(f'KDE of Coupling Avg. Error for {dataset}')
        ax1.set_xlabel('Coupling Avg. Error')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        ax2 = axes[1]
        plot_data = []
        labels = []

        for algo in algorithms:
            algo_data = dataset_data[dataset_data['name'] == algo]['coupling_avg_err']
            if len(algo_data) > 0:
                plot_data.append(algo_data)
                labels.append(algo)

        ax2.boxplot(plot_data, tick_labels=labels, showmeans=True)
        ax2.set_title(f'Box Plot of Coupling Avg. Error for {dataset}')
        ax2.set_ylabel('Coupling Avg. Error')
        ax2.set_yscale('log')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{dataset}_coupling_err_distributions.png"))
        plt.close(fig)


for group_name, datasets in dataset_groups.items():
    print(f"\n### {group_name} Runtime Analysis ###")
    plot_runtime_distributions(datasets, results_df, algorithms, save_path='runtime_distributions')
    print(f"\n### {group_name} Cost Error Analysis ###")
    plot_cost_err_distributions(datasets, results_df, algorithms, save_path='cost_err_distributions')
    print(f"\n### {group_name} Coupling Error Analysis ###")
    plot_coupling_err_distributions(datasets, results_df, algorithms, save_path='coupling_err_distributions')

import os
import matplotlib.pyplot as plt
from IPython.display import HTML
import dataframe_image as dfi

def save_analysis_tables_by_dimension(results_df, metrics, output_dir="analysis_tables"):
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, datasets in dataset_groups.items():
        dim_folder = os.path.join(output_dir, group_name.replace(" ", "_"))
        os.makedirs(dim_folder, exist_ok=True)
        
        dim_data = results_df[results_df['dataset'].isin(datasets)]
        
        for metric in metrics:
            mean_table = get_mean_comparison_table(dim_data, metric)
            mean_file = os.path.join(dim_folder, f"{metric}_mean.csv")
            mean_table.to_csv(mean_file, index=False)
            
            std_table = get_std_comparison_table(dim_data, metric)
            std_file = os.path.join(dim_folder, f"{metric}_std.csv")
            std_table.to_csv(std_file, index=False)
            
            def highlight_top_two(s):
                is_first = s == s.min()
                is_second = s == s[s != s.min()].min() if len(s[s != s.min()]) > 0 else pd.Series(False, index=s.index)
                return [
                    'background-color: lightgreen' if f else
                    'background-color: lightblue' if s_ else
                    ''
                    for f, s_ in zip(is_first, is_second)
                ]
            
            mean_styled = mean_table.style.apply(highlight_top_two, axis=1, subset=mean_table.columns[1:])
            mean_html_file = os.path.join(dim_folder, f"{metric}_mean.html")
            with open(mean_html_file, 'w') as f:
                f.write(mean_styled.to_html())
            
            mean_img_file = os.path.join(dim_folder, f"{metric}_mean.png")
            dfi.export(mean_styled, mean_img_file)
                
            std_styled = std_table.style.apply(highlight_top_two, axis=1, subset=std_table.columns[1:])
            std_html_file = os.path.join(dim_folder, f"{metric}_std.html")
            with open(std_html_file, 'w') as f:
                f.write(std_styled.to_html())
            
            std_img_file = os.path.join(dim_folder, f"{metric}_std.png")
            dfi.export(std_styled, std_img_file)
                
        print(f"Saved analysis tables for {group_name} to {dim_folder}")

save_analysis_tables_by_dimension(results_df, metrics_to_analyze)