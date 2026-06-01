import numpy as np
import torch
from models.vae import HVAE_Lightning2
from dataset.singlecell import SingleCellDataset
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
def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    

def save_validate(dataloader, model, save_embedding):
    cuda_id = 0
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
        
    np.savez(save_embedding, embedding = mu_list, batches = np.array(batched_data_list), labels = np.array(label_data_list))

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
    
    
    
    #train_dataset = SingleCellDataset(train_data_name, gene_names, dataset_symbol, label_symbols, data_symbol, gene_symbol, count_symbol = count_symbol)
    
    #if val_data_path != None:
    #    val_dataset = SingleCellDataset(val_data_path)
    #print(train_dataset.return_size())
    #print(train_dataset.return_num_batches())


    data_manager = DataLoaderManager(
            domain_key="dataset", 
            class_key=None, 
            covariate_keys=None, 
            feature_list=None,
            normalize_total=False,
            log1p=False,    
            batch_size=32, 
            train_frac=1.0,
            use_sampler=True,
            sampler_emb="X_pca",
            sampler_knn=10,
            sampler_domain_minibatch_strategy='proportional',
            domain_coverage=None,
            dist_metric='euclidean', 
            p_intra_knn=0.3, 
            p_intra_domain=1.0, 
            use_faiss=True, 
            use_ivf=True, 
            ivf_nprobe=10, 
            preload_dense=False,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )
        
    train_dataloader, val_dataloader, data_structure, info = data_manager.anndata_to_dataloader(train_adata)
    batches_out,data_out,size_, num_batches, number_marker_genes = info

    model = HVAE_Lightning2(n_input=size_, n_latent=latent_dim,n_hidden = hidden_dim, n_layers=layers, 
                            encode_covariates=encode_covariates, ln = ln, num_components = num_batches, 
                            gene_factor = gene_factor, prediction_genes = number_marker_genes, 
                            n_batch = num_batches, random_assignment = random_assignment, kl_weight = kl_weight, 
                            mmd_weight = mmd_weight, reconstruction_weight = reconstruction_weight,
                            kl_anneal_warmup_epochs = kl_anneal_warmup_epochs, kl_anneal_max = kl_anneal_max, weights = None)
    model.to(device)
    model.get_dataset_baseline(data_out, batches_out, n_batch = num_batches)



    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    dataloader = train_dataloader
    #val_dataloader = train_dataloader

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
    trainer.fit(model, dataloader, val_dataloader)
    trainer.save_checkpoint(model_checkpoint)
    
    model.to("cpu")
    save_validate(val_dataloader, model, save_embedding)
    
    

        
if __name__ == "__main__":


    cmdline_parser = argparse.ArgumentParser('run hvae')


    cmdline_parser.add_argument('-d', '--dataset',
                                help='dataset',
                                default = '../data/test_query.npz',
                                required = True,
                                type=str)
    cmdline_parser.add_argument('-s', '--dataset_symbol',
                                help='dataset',
                                default = 'batch',
                                type=str)
    cmdline_parser.add_argument('-l', '--label_symbols',
                                help='cell_subtype',
                                default = 'batch',
                                type=str)
    cmdline_parser.add_argument('-e', '--data_symbol',
                                help='data_symbol',
                                default = 'data',
                                type=str)
    cmdline_parser.add_argument('-g', '--gene_symbol',
                                help='gene_names',
                                default = 'gene_names',
                                type=str)
    cmdline_parser.add_argument('-o', '--output',
                                help='output embedding',
                                default = 'embedding.npz',
                                type=str)
    cmdline_parser.add_argument('-m', '--model',
                                help='output model',
                                default = 'model.ckpt',
                                type=str)
    cmdline_parser.add_argument('-e', '--epochs',
                                help='number of epochs',
                                default = 50,
                                type=int)
    cmdline_parser.add_argument('-z', '--latent_dimension',
                                help='latent_dimension',
                                default = 90,
                                type=int)
    cmdline_parser.add_argument('-k', '--hidden_dimension',
                                help='hidden_dimension',
                                default = 128,
                                type=int)
    cmdline_parser.add_argument('-c', '--layers',
                                help='mumber of layers',
                                default = 1,
                                type=int)
    cmdline_parser.add_argument('-r', '--learning_rate',
                                help='learning rate',
                                default = 0.0001,
                                type=float)
    cmdline_parser.add_argument('-b', '--batch_size',
                                help='batch size',
                                default = 255,
                                type=int)
    cmdline_parser.add_argument('-p', '--patience',
                                help='patience',
                                default = 15,
                                type=int)
    cmdline_parser.add_argument('-y', '--gene_factor',
                                help='weight for marker genes',
                                default = 20,
                                type=int)
    cmdline_parser.add_argument('-z', '--marker_genes',
                                help='path to marker gene list',
                                default = "marker_genes.txt",
                                type=str)
                                
    args, unknowns = cmdline_parser.parse_known_args()

    set_seed()    
    gene_names = open(args.marker_genes).readlines()
    
    
    
    main(train_data_name = args.dataset, gene_names = gene_names, dataset_symbol = args.dataset_symbol,
          label_symbols = args.label_symbols,  data_symbol = args.data_symbol, gene_symbol = args.gene_symbol, 
            val_data_path = None, num_epochs = 50, latent_dim = 300, hidden_dim = 128, layers = 1, learning_rate = 0.0001, 
            batch_size = 255, patience = 15, encode_covariates =True, ln = False, gene_factor = args.gene_factor, kl_weight = args.kl_weight, 
            mmd_weight = args.mmd_weight, reconstruction_weight = args.reconstruction_weight, save_embedding = args.embedding_name,
              kl_anneal_warmup_epochs = args.kl_anneal_warmup_epochs, kl_anneal_max = args.kl_anneal_max, model_checkpoint = "model.ckpt", count_symbol = None, random_assignment =  False)
    
