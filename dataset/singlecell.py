import torch
import numpy as np
from torch.utils.data import Dataset

def create_bins(data, n_bins = 10):

    bin_edges = np.linspace(np.min(data), np.max(data), n_bins + 1)
    bin_indices = np.digitize(data, bin_edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
    return bin_indices/10
    
class SingleCellDataset(Dataset):
    def __init__(self, data_path, gene_names, dataset_symbol, label_symbols, data_symbol, gene_symbol, 
    load_raw_indices=False, sample_fraction = 1.0, return_label_ints = False, count_symbol = None):

        processed_data = np.load(data_path, allow_pickle=True)
        self.data = processed_data[data_symbol]
        if count_symbol == None: self.count_data = np.log1p(self.data)
        else: self.count_data = processed_data[count_symbol]
        self.label = processed_data[label_symbols]
        unique_labels = np.unique(self.label)
        self.label_int = [list(unique_labels).index(l) for l in self.label]
        self.label = [str(l) for l in self.label]
        self.batch = processed_data[dataset_symbol]
        
        self.batch_dict = list(np.unique(self.batch))
        self.batch = [self.batch_dict.index(a) for a in self.batch]
        
        gene_data = processed_data[gene_symbol]
        
        bin_indices = []
        self.prediction_genes = 0
        
        if return_label_ints:
            self.label = self.label_int
        
        
        for gene_sym in gene_names:
            try:
                relevant_int = list(gene_data).index(gene_sym)
                relevant_data = self.count_data[:,relevant_int]
                bin_data = np.expand_dims(create_bins(relevant_data, n_bins = 10),axis=-1)
                bin_indices.append(bin_data)
                self.prediction_genes = self.prediction_genes + 1
            except:
                print("skipped: " + str(gene_sym))
                
        
        self.bin_indices = np.concat(bin_indices, axis = -1)

        if sample_fraction < 1.0:
            indices = np.random.choice(len(self.data), int(sample_fraction * len(self.data)), replace=False)
            self.data = self.data[indices]
            self.label = np.array(self.label)[indices]
            self.bin_indices = np.array(self.bin_indices)[indices]
        

    def __len__(self):
        return len(self.data)

    def return_size(self):
        return np.shape(self.data)[1]
        
    def num_marker_genes(self):
        return self.prediction_genes
        
    def return_num_batches(self):
        return len(self.batch_dict)
    
    def return_data(self):
        return self.data
    
    def return_batches(self):
        return self.batch
        
        
    def __getitem__(self, index):
        return (
            self.data[index],
            self.label[index],
            self.batch[index],
            self.bin_indices[index],
        )
        

