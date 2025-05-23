import os
import argparse
import subprocess
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Visualize experiment results.")
parser.add_argument(
    "--results-dir",
    type=str,
    required=True,
    help="Path to the directory containing experiment results."
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

    ranks_part = ranks_part.split('\n') 
    
    ranks_algorithms = ranks_part[0].split()
    ranks = ranks_part[1].split()
    ranks = pd.Series(index=ranks_algorithms, data=ranks)

    return pvalues, ranks


def create_graph_for(pvalues: pd.DataFrame, ranks: pd.Series, significance = 0.05):    
    G = nx.Graph()

    for i, algorithm in enumerate(ranks.index):
        G.add_node(algorithm, name=algorithm, info=f"Rank: {ranks[algorithm]}")

    for first_algorithm in ranks.index:
        for second_algorithm in ranks.index:
            pvalue = pvalues.loc[first_algorithm, second_algorithm]
            if pvalue < significance:
                G.add_edge(first_algorithm, second_algorithm)

    pos = {node: (0, -i) for i, node in enumerate(G.nodes())}

    nx.draw_networkx_edges(G, pos)

    ax = plt.gca()
    for node in G.nodes(data=True):
        node_id, attrs = node
        name = attrs.get("name", "")
        info = attrs.get("info", "")
        label = f"{name}\n{info}"

        x, y = pos[node_id]

        ax.text(x, y, label,
                fontsize=10,   
                ha='center', va='center',
                bbox=dict(fc="lightblue", ec="black", lw=1),
                linespacing=1.2)
    
    plt.axis('off')
    plt.show()



result_files = [os.path.join(args.results_dir, file) for file in os.listdir(args.results_dir)]
test_results = [parse_post_hoc_result(file) for file in result_files]

create_graph_for(*test_results[0])


