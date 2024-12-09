import torch
import numpy as np
from torch.utils.data import Dataset


class SingleCellDataset(Dataset):
    def __init__(self, data_path, load_raw_indices=False):
        processed_data = np.load(data_path, allow_pickle=True)
        self.data = processed_data["data"]
        self.labels = processed_data["labels"]
        self.genes = processed_data["genes"]

        if load_raw_indices:
            self.raw_indices = processed_data["raw_indices"]

        self.label_to_index = {
            label: index for index, label in enumerate(sorted(set(self.labels)))
        }
        self.label_indices = [self.label_to_index[label] for label in self.labels]

        #####
        # self.organs = processed_data["organs"]
        # self.organ_to_index = {
        #     organ: index for index, organ in enumerate(sorted(set(self.organs)))
        # }
        # self.organ_indices = [self.organ_to_index[organ] for organ in self.organs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.data[index],
            self.label_indices[index],
        )  # , self.organ_indices[index]
