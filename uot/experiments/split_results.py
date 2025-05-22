import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Process result file path.")
parser.add_argument('--result-file', type=str, required=True, help='Path to the result file')
parser.add_argument('--export-dir', type=str, required=True, help='Path to the export dir')
args = parser.parse_args()

df = pd.read_csv(args.result_file).drop(columns=['Unnamed: 0'])
aggregation_columns = list(df.columns.difference(["time", 'cost_rerr', "coupling_avg_err", 'converged', 'epsilon', 'name']))
df['problem_id'] = df[aggregation_columns].apply(lambda row: '|'.join([str(x) for x in row if pd.notna(x)]), axis=1)

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

base_filename = os.path.splitext(os.path.basename(args.result_file))[0]

time_df.to_csv(os.path.join(args.export_dir, f"{base_filename}_time.csv"))
cost_rerr_df.to_csv(os.path.join(args.export_dir, f"{base_filename}_cost_rerr.csv"))
couplind_avg_err.to_csv(os.path.join(args.export_dir, f"{base_filename}_coupling_avg_err.csv"))