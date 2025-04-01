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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.data[index],
            self.label_indices[index],
        )  # , self.organ_indices[index]
        
        
class SingleCellDataset2(Dataset):
    def __init__(self, data_path, load_raw_indices=False):

        processed_data = np.load(data_path, allow_pickle=True)
        self.return_count = 0
        self.data = processed_data['data']
        self.batch = processed_data['batch']
        self.batch = [0 if a == "dataset1" else 1 for a in self.batch]

        labels = processed_data["label"]
        labels = [str(l) for l in labels]
        labels2 = processed_data["label2"]
        labels2 = [str(l) for l in labels2]
        mapping = {'pDCs': 'pDC'}
        labels2 = [mapping[l] if l == 'pDCs' else l for l in labels2]
        labels = [l if l != "nan" else labels2[enum] for enum, l in enumerate(labels)]
        labels_low = processed_data["label_low"]
        labels_low = [str(l) for l in labels_low]
        labels_low = [l if l != "nan" else labels2[enum] for enum, l in enumerate(labels_low)]
        label_curated = processed_data["label_curated"]
        label_curated = [str(l) for l in label_curated]
        label_curated = [l if l != "nan" else labels2[enum] for enum, l in enumerate(label_curated)]
        label_predicted = processed_data["label_predicted"]
        label_predicted = [str(l) for l in label_predicted]
        label_predicted = [l if l != "nan" else labels2[enum] for enum, l in enumerate(label_predicted)]
        label_cathegory = processed_data["label_cathegory"]
        label_cathegory = [str(l) for l in label_cathegory]
        label_cathegory = [l if l != "nan" else labels2[enum] for enum, l in enumerate(label_cathegory)]
        
        
        self.label = np.array( [label + "+" + labels_low[enum] + "+" + label_curated[enum] + "+" + label_predicted[enum] + "+" + label_cathegory[enum] for enum, label in enumerate(labels)])


        if load_raw_indices:
            self.raw_indices = processed_data["raw_indices"]

        self.label_to_index = {
            label: index for index, label in enumerate(sorted(set(self.batch)))
        }

        self.label_indices = [self.label_to_index[label] for label in self.batch]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.return_count = self.return_count + 1
        return (
            self.data[index],
            self.label[index],
            self.return_count,
            self.batch[index]
        ) 
