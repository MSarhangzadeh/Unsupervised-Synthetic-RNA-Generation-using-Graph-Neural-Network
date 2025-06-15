import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import shutil
from utils import load_and_filter_dataset, one_hot_encode_sequence, create_graph_from_sequence


class PiRNADataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for piRNA sequences as graphs.
    """
    def __init__(self, root, df, transform=None, pre_transform=None):
        self.df = df
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        return ['filtered_piRNAs.csv']
    
    @property
    def processed_file_names(self):
        return ['piRNA_graphs.pt']
    
    def process(self):
        print("Processing dataset...")
        data_list = []
        for idx, row in self.df.iterrows():
            sequence = row['seq']
            graph = create_graph_from_sequence(sequence)
            graph.y = torch.tensor([row['len']], dtype=torch.float)
            data_list.append(graph)
        
        print(f"Created {len(data_list)} graphs.")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Dataset saved to {self.processed_paths[0]}")