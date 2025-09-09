import torch
import numpy as np
from torch.utils.data import Dataset


class SingleCellDataset(Dataset):
    def __init__(self, data_path, batch_dict, gene_names, dataset_symbol, label_symbols, data_symbol, gene_symbol, load_raw_indices=False, sample_fraction = 1.0):

        processed_data = np.load(data_path, allow_pickle=True)
        self.data = processed_data[data_symbol]
        self.label = processed_data[label_symbols]
        self.label = [str(l) for l in self.label]
        self.batch = processed_data[dataset_symbol]
        self.batch = [batch_dict[a] for a in self.batch]
        
        gene_names = processed_data[gene_symbol]
        
        bin_indices = []
        self.prediction_genes = 0
        
        for gene_sym in gene_names:
            try:
                relevant_int = list(gene_names).index(gene_sym)
                relevant_data = self.data[:,relevant_int]
                bin_data = np.expand_dims(create_bins(relevant_data, n_bins = 10),axis=-1)
                bin_indices.append(bin_data)
                self.prediction_genes = self.prediction_genes + 1
            except:
                print("skipped: " + str(gene_sym))
                
        
        self.bin_indices = np.concat(bin_indices, axis = -1)

        if sample_fraction <= 1.0:
            indices = np.random.choice(len(self.data), int(sample_fraction * len(self.data)), replace=False)
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.bin_indices = self.bin_indices[indices]
        

    def __len__(self):
        return len(self.data)

    def return_size(self):
        return np.shape(self.data)[0]
        
    def num_marker_genes(self):
        return self.prediction_genes
        
    def return_num_batches(self):
        return len(batch_dict.keys())

    def __getitem__(self, index):
        return (
            self.data[index],
            self.label[index],
            self.batch[index],
            self.bin_indices[index],
        )
        

#batch_dict = {'Che_et_al':0, 'Durante_et_al':1, 'Lin_et_al':2, 'Lin_k_et_al':3, 'Rao_et_al':4,
# 'Sathe_et_al':5, 'Wang_et_al':6, 'Zhang_et_al':7, 'liverAtlas':8}

 #       gene_symbol = ["PTPRC", "CD2", "CD3D","CD3G","CD3E", "CD8A","CD8B", "CD4", "MZB1", "CD19", "MS4A1", "CD14", "FCGR3A", "FCGR1A", "ITGAM", "CD68", "SPP1", "IL3RA",  "C1QA", "TYROBP", "FCER1G", "CPA3",  "TPSAB1", "MS4A2", "KIT", "ACTA2", "THY1", "CRABP1", "VWF", "ENG", "CDH5", "VCAM1", "KRT19", "KRT8", "EPCAM", "ALB", "TF", "TTR", "MLANA", "TYRP1", "MITF", "PMEL", "FDCSP", "SPIB", "ASCL1", "CALCA", "CHGA", "CHGB", "TFF3","GRP", "CEACAM5", "CEACAM6", "SCGB2A2","ENO2","SSTR2"]
