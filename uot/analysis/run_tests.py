import os
import argparse
import subprocess
import pandas as pd

parser = argparse.ArgumentParser(description="Visualize experiment results.")
parser.add_argument(
    "--results-dir",
    type=str,
    required=True,
    help="Path to the directory containing experiment results."
)
parser.add_argument(
    "--export-dir",
    type=str,
    required=True,
    help="Path to the directory where output tables will be saved."
)

args = parser.parse_args()
results_dir = args.results_dir


def parse_post_hoc_result(filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = subprocess.run(['Rscript', 'uot/experiments/post_hoc_test.R', filename], capture_output=True, text=True)
    output = result.stdout

    pvalues_part, ranks_part = output.split("\n\n")

    pvalues_part = pvalues_part.split("\n")

    algorithms = pvalues_part[0].split()
    
    pvalues = [row.split()[1:] for row in pvalues_part[1:]]
    pvalues = pd.DataFrame(columns=algorithms, index=algorithms, data=pvalues)
    pvalues = pvalues.apply(pd.to_numeric, errors='coerce')

    ranks_part = ranks_part.split('\n')[1:-1]
    
    ranks_algorithms = [row.split()[0] for row in ranks_part]
    ranks = [row.split()[1] for row in ranks_part]
    ranks = pd.Series(index=ranks_algorithms, data=ranks)

    return pvalues, ranks


def convert_to_latex_tables(pvalues: pd.DataFrame, ranks: pd.Series) -> None:
    table_code = pvalues.to_latex(column_format=f"|l|{'l'*len(pvalues)}|")

    table_code = r"\begin{tabular}{" + f"|l|{'l'*len(pvalues)}|" + '}\n'
    table_code += r'\hline' + '\n'
    table_code += ' & '.join([' '] + list(pvalues.columns)) + r'\\' + '\n'
    table_code += r'\hline' + '\n'

    for algorithm in pvalues.index:
        row_items = map(str, [algorithm] + list(pvalues.loc[algorithm, :]))
        table_code += ' & '.join(map(str, row_items)) + r'\\' + '\n'

    table_code += r'\hline\end{tabular}'
    ranks = ranks.sort_values().to_frame("Rank")
    return pvalues.to_latex(float_format="%.2e"), ranks.to_latex()


result_files = [os.path.join(args.results_dir, file) for file in os.listdir(args.results_dir)]

for filepath in result_files:
    pvalues, ranks = parse_post_hoc_result(filepath)
    pvalues_table, ranks_table = convert_to_latex_tables(pvalues, ranks)

    basename = os.path.splitext(os.path.basename(filepath))[0]
    basename = f"{basename}_summary" 

    with open(os.path.join(args.export_dir, basename), 'w', encoding='utf8') as output_file:
        output_file.write("P-values Latex table:\n")
        output_file.write(pvalues_table)
        output_file.write("\n")
        output_file.write("Rank Latex table:\n")
        output_file.write(ranks_table)


