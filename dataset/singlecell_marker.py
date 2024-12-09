import torch
import numpy as np
from torch.utils.data import Dataset


class SingleCellDataset(Dataset):
    def __init__(self, data_path, markers_name, load_raw_indices=False):
        processed_data = np.load(data_path, allow_pickle=True)
        self.data = processed_data["data"]
        self.labels = processed_data["labels"]
        self.genes = processed_data["genes"]
        self.marker_genes_indices = processed_data[markers_name]

        if load_raw_indices:
            self.raw_indices = processed_data["raw_indices"]

        self.label_to_index = {
            label: index for index, label in enumerate(sorted(set(self.labels)))
        }
        self.label_indices = [self.label_to_index[label] for label in self.labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.data[index, self.marker_genes_indices],
            self.label_indices[index],
        )
