import numpy as np
import torch
from models.vae import HVAE_Lightning
import logging
import time
import os
import json
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from pytorch_lightning import Trainer
import random
from pytorch_lightning.callbacks import EarlyStopping
import argparse
from dataset.dataloader import DataLoaderManager
import scanpy as sc


def save_validate(dataloader, model, save_embedding):
    device = "cpu"
    
    mu_list = []
    batched_data_list = []
    label_data_list = []
    data_list = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            data, label_data,batched_data, gene = batch_data
            data = data.to(device).float()

            batched_data_list.extend(np.array(batched_data))
            label_data_list.extend(np.array(label_data))
            batched_data = batched_data.to(device)
            output = model(data, gene, batched_data)
            data_list.extend(data.to("cpu"))
            mu = output["z_mu"]
            mu_list.extend(mu.detach().cpu().numpy())
        
    np.savez(save_embedding, embedding = mu_list, batches = np.array(batched_data_list),
             labels = np.array(label_data_list))

    return 


        

        
def main(train_data_name, gene_names, dataset_symbol, label_symbols, 
         data_symbol, gene_symbol, val_data_path = None, 
         num_epochs = 2, latent_dim = 90, hidden_dim = 128, 
         layers = 1, learning_rate = 0.0001, batch_size = 256,
           patience = 15, encode_covariates =True, ln = False, 
           gene_factor = 20, kl_weight = 0.01, 
            mmd_weight = 0.5, reconstruction_weight = 1.0, 
            kl_anneal_warmup_epochs = 20, kl_anneal_max = 0.5, 
             save_embedding = "embedding.npz",
             model_checkpoint = "model.ckpt", count_symbol = None, random_assignment = False):


    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    train_adata = sc.read(os.path.join(train_data_name))

    data_manager = DataLoaderManager(
            domain_key=dataset_symbol,
            class_key=None, 
            covariate_keys=None, 
            feature_list=None,
            normalize_total=False,
            log1p=False,    
            batch_size=33,
            train_frac=1.0,
            use_sampler=True,
            sampler_emb=None,
            sampler_knn=None,
            sampler_domain_minibatch_strategy='proportional',
            domain_coverage=None,
            dist_metric='euclidean', 
            p_intra_knn=0.0,
            p_intra_domain=0.7,
            use_faiss=True, 
            use_ivf=True, 
            ivf_nprobe=10, 
            preload_dense=False,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )



    train_dataloader, val_dataloader, data_structure, info = data_manager.anndata_to_dataloader(train_adata, dataset_symbol,gene_names)
    batches_out,data_out,size_, num_batches, number_marker_genes = info

    print(train_dataloader)
    print(val_dataloader)

    model = HVAE_Lightning(n_input=size_, n_latent=latent_dim,n_hidden = hidden_dim, n_layers=layers,
                            encode_covariates=encode_covariates, ln = ln, num_components = num_batches, 
                            gene_factor = gene_factor, prediction_genes = number_marker_genes, 
                            n_batch = num_batches, random_assignment = random_assignment, kl_weight = kl_weight, 
                            mmd_weight = mmd_weight, reconstruction_weight = reconstruction_weight,
                            kl_anneal_warmup_epochs = kl_anneal_warmup_epochs, kl_anneal_max = kl_anneal_max)
    model.to(device)
    model.get_dataset_baseline(data_out, batches_out, n_batch = num_batches)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    early_stop_callback = EarlyStopping(
    monitor="train_loss",  
    patience=patience,          
    mode="min",            
    verbose=True
    )

    trainer = Trainer(
    max_epochs=num_epochs, 
    callbacks=[early_stop_callback],
    precision = 64,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint(model_checkpoint)
    
    model.to("cpu")
    save_validate(val_dataloader, model, save_embedding)
