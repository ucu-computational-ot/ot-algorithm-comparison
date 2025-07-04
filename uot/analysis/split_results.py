import os
import argparse
import pandas as pd


def split_result_file(result_filepath: str, export_dir: str):
    df = pd.read_csv(result_filepath).drop(columns=['Unnamed: 0'])

    aggregation_columns = [column for column in df.columns if column.startswith('source_') or column.startswith('target_')]
    solver_specification_solumns = df.columns.difference(aggregation_columns + ['time', 'cost_rerr', 'coupling_avg_err', 'converged', 'dataset', 'name'])

    df['problem_id'] = df[aggregation_columns].apply(lambda row: '|'.join([str(x) for x in row if pd.notna(x)]), axis=1)
    df['solver_kwargs'] = pd.Series(df[solver_specification_solumns].to_dict(orient='records'))
    df['name'] = df.apply(lambda row: f"{row['name']}({row['solver_kwargs']})" , axis=1)
    df['name'] = df['name'].str.replace(' ', '')

    df = df.groupby(['problem_id', 'name']).agg({
        'time': 'mean',
        'cost_rerr': 'mean',
        'coupling_avg_err': 'mean',
    }).reset_index()

    time_df = df[['problem_id', 'time', 'name']]
    cost_rerr_df = df[['problem_id', 'cost_rerr', 'name']]
    couplind_avg_err = df[['problem_id', 'coupling_avg_err', 'name']]

    time_df = time_df.pivot(index='problem_id', columns='name', values='time')
    cost_rerr_df = cost_rerr_df.pivot(index='problem_id', columns='name', values='cost_rerr')
    couplind_avg_err = couplind_avg_err.pivot(index="problem_id", columns="name", values="coupling_avg_err")

    base_filename = os.path.splitext(os.path.basename(result_filepath))[0]

    time_df.to_csv(os.path.join(export_dir, f"{base_filename}_time.csv"))
    cost_rerr_df.to_csv(os.path.join(export_dir, f"{base_filename}_cost_rerr.csv"))
    couplind_avg_err.to_csv(os.path.join(export_dir, f"{base_filename}_coupling_avg_err.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process result file path.")
    parser.add_argument('--result-file', type=str, required=True, help='Path to the result file')
    parser.add_argument('--export-dir', type=str, required=True, help='Path to the export dir')
    args = parser.parse_args()
    split_result_file(result_filepath=args.result_file, export_dir=args.export_dir)

