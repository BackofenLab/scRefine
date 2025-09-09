import os
import tempfile
import numpy as np
import scanpy as sc
import seaborn as sns
import pandas as pd


def translate(data_dir, dataset, npy_dataset, obs_fields, layer_fields):

    adata_ref = sc.read(os.path.join(data_dir, dataset))
    
    output = {}
    output["data"] = adata_ref.X.toarray()
    output["gene_names"] = np.array(adata_ref.var_names)
    
    for obs in obs_fields:
        output[obs] = adata_ref.obs[obs]
    for obs in layer_fields:
        output[obs] = adata_ref.layers[obs]
    
    np.savez(data_dir + npy_dataset, **output)

if __name__ == '__main__':

    random.seed(42)
    translate(data_dir = "../data/", dataset = "test_ref.h5ad", npy_dataset = "test_ref.npz", obs_fields = ["NormalvsTumor","SampleID","doublet_score","Tissue"], layer_fields = ["counts"])
