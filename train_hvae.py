import numpy as np
import torch
from models.vae import HVAE_Lightning7
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
        
    np.savez(save_embedding, embedding = mu_list, batches = np.array(batched_data_list), labels = np.array(label_data_list), or_data = np.array(data_list))

    return 


        
def main(train_data_name, gene_names, dataset_symbol, label_symbols, data_symbol, gene_symbol, val_data_path = None, num_epochs = 50, latent_dim = 90, hidden_dim = 128, layers = 1, learning_rate = 0.0001, batch_size = 256, patience = 15, encode_covariates =True, ln = False, gene_factor = 20, save_embedding = "embedding.npz", model_checkpoint = "model.ckpt"):


    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = SingleCellDataset(train_data_name, gene_names, dataset_symbol, label_symbols, data_symbol, gene_symbol)
    
    if val_data_path != None:
        val_dataset = SingleCellDataset(val_data_path)
    
    model = HVAE_Lightning7(n_input=train_dataset.return_size(), n_latent=latent_dim,n_hidden = hidden_dim, n_layers=layers, encode_covariates=encode_covariates, ln = ln, num_components = train_dataset.return_num_batches(), gene_factor = gene_factor, prediction_genes = train_dataset.num_marker_genes(), n_batch = train_dataset.return_num_batches(), random_assignment = False)
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    if val_data_path != None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
    else:
        val_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )

    early_stop_callback = EarlyStopping(
    monitor="train_loss",  
    patience=patience,          
    mode="min",            
    verbose=True
    )

    trainer = Trainer(
    max_epochs=num_epochs, 
    callbacks=[early_stop_callback]
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
    
    
    main(train_data_name = args.dataset, gene_names = gene_names, dataset_symbol = args.dataset_symbol, label_symbols = args.label_symbols,  data_symbol = args.data_symbol, gene_symbol = args.gene_symbol,  val_data_path = None, num_epochs = args.epochs, latent_dim = args.latent_dimension, hidden_dim = args.hidden_dimension, layers = args.layers, learning_rate = args.learning_rate, batch_size = args.batch_size, patience = args.patience, encode_covariates =True, ln = False, gene_factor = args.gene_factor, save_embedding = args.output, model_checkpoint = args.model)
    
