import torch
import numpy as np
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, data_path):
        processed_data = np.load(data_path, allow_pickle=True)
        self.data = processed_data["data"]
        self.labels = processed_data["labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
