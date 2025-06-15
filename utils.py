import os
import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data,  InMemoryDataset

import networkx as nx
import matplotlib.pyplot as plt


def load_and_filter_dataset(file_path, save_filtered=False, output_path="data/filtered_piRNAs.csv"):
    """
    Loads and filters a piRNA dataset from a tab-delimited CSV file.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only piRNAs of length 26–32.
    """
    df = pd.read_csv(file_path, sep=None, header=None)
    df.columns = ['rnaClass', 'rnaID', 'chr', 'chrStart', 'chrEnd', 'strand', 'len', 'seq', 'entryID']

    df['len'] = pd.to_numeric(df['len'], errors='coerce')

    if df['len'].isna().any():
        print(f"Warning: Found {df['len'].isna().sum()} non-numeric values in 'len' column. These rows will be excluded from filtering.")

    total_entries = len(df)
    unique_classes = df['rnaClass'].nunique()
    class_counts = df['rnaClass'].value_counts()
    print("Dataset Summary (Before Filtering):")
    print(f"  Total entries: {total_entries}")
    print(f"  RNA types: {unique_classes}")
    print(class_counts)
    
    print("-----------------------------------------")

    filtered_df= df[(df['rnaClass'] == 'piRNA') & (df['len'].between(26, 32))].copy()

    print("Filtered Dataset:")
    print(f"  Total piRNAs with length 26–32: {len(filtered_df)}")
    print(f"  Unique piRNA IDs: {filtered_df['rnaID'].nunique()}")
    print(f"  Length range in filtered data: {filtered_df['len'].min()} - {filtered_df['len'].max()}")

    if save_filtered:
        filtered_df.to_csv(output_path, index=False)
        print(f"Filtered data saved to {output_path}")

    return filtered_df



def one_hot_encode_sequence(sequence, max_len=32):
    """
    Convert a piRNA sequence to a [32 x 5] one-hot encoded matrix.
    Pad with X ([0,0,0,0,1]) for sequences shorter than 32.

    Returns:
        np.ndarray: [32 x 5] one-hot encoded matrix.
    """
    # Nucleotide to one-hot mapping
    nucleotide_map = {
        'A': [1, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0],
        'G': [0, 0, 1, 0, 0],
        'U': [0, 0, 0, 1, 0],
        'X': [0, 0, 0, 0, 1]
    }
    
    sequence = sequence.upper()
    
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        sequence = sequence + 'X' * (max_len - len(sequence))
    
    one_hot = np.array([nucleotide_map.get(nt, [0, 0, 0, 0, 1]) for nt in sequence], dtype=np.float32)
    
    return one_hot # output -> 32*5


    

def create_graph_from_sequence(sequence, max_len=32):
    """
    Create a PyTorch Geometric graph from a piRNA sequence.
    
    Returns:
        Data: PyTorch Geometric Data object with node features and edges.
    """
    node_features = one_hot_encode_sequence(sequence, max_len)
    
    edge_index = []
    for i in range(max_len - 1):
        edge_index.append([i, i + 1])  
        edge_index.append([i + 1, i])  # to have both forward and backward edge to have shap of (2,x)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index
    )
    
    return data



def plot_graph(data, output_path="graph.png"):
    """
    Plot the PyTorch Geometric graph using NetworkX and Matplotlib.

    """
    nucleotides = ['A', 'C', 'G', 'U', 'X']
    node_labels = [nucleotides[int(torch.argmax(row))] for row in data.x]
    
    G = nx.Graph()
    for i, label in enumerate(node_labels):
        G.add_node(i, label=label)
    edges = data.edge_index.t().tolist()
    G.add_edges_from([(e[0], e[1]) for e in edges if e[0] < e[1]])  
    pos = {i: (i, 0) for i in range(len(node_labels))}  
    
    # Plot the graph
    plt.figure(figsize=(15, 2))
    nx.draw(G, pos, with_labels=True, labels={i: G.nodes[i]['label'] for i in G.nodes},
            node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    plt.title("piRNA Sequence Graph")
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    print(f"Graph plot saved to {output_path}")

