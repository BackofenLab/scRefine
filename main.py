import numpy as np
import torch
from models.encoder import Encoder
from dataset.singlecell import SingleCellDataset
import logging
import time
import os
import json
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset
import argparse
from train_hvae import main
from preprocessing.translate_data import  translate
from preprocessing.create_data import  preprocessing
from clean_data import clean_up
import random
import numpy as np
import torch


def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="handles data preprocessing, traning and clean up")
    
    #### pre-processing arguments #####
    parser.add_argument("--dataset_path", type=str, help="path to dataset")
    parser.add_argument("--marker_genes", type=str, default="marker_genes.txt", help="path to marker genes")
    parser.add_argument("--output_h5ad", type=str, default="output", help="preprocessing output h5ad")
    parser.add_argument("--npy_dataset", type=str, default="output.npz", help="output npy dataset")
    parser.add_argument("--dataset_symbol", type=str, default="batch", help="key to be used for batch correction")
    parser.add_argument("--no_hvg", action=argparse.BooleanOptionalAction, default = False)
    #### training arguments #####
    parser.add_argument("--label_symbols", type=str, default="batch", help="labels used for plotting")
    parser.add_argument("--data_symbol", type=str, default="data", help="data field")
    parser.add_argument("--gene_symbol", type=str, default="gene_names", help="gene names")
    parser.add_argument("--count_symbol", type=str, default=None, help="normalized data")
    parser.add_argument("--output", type=str, default="embedding.npz", help="output embedding")
    parser.add_argument("--model", type=str, default="model.ckpt", help="output model")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--latent_dimension", type=int, default=90, help="size of latent dim")
    parser.add_argument("--hidden_dimension", type=int, default=128, help="size of hidden dim")
    parser.add_argument("--layers", type=int, default=1, help="mumber of layers")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=255, help="batch size")
    parser.add_argument("--patience", type=int, default=15, help="patience")
    parser.add_argument("--gene_factor", type=int, default=50, help="weight for marker genes")
    parser.add_argument("--encode_covariates", type=bool, default=True, help="encoding covariates")
    parser.add_argument("--ln", type=bool, default=False, help="layer normalization")
    parser.add_argument("--val_data_path", type=str, default=None, help="alternative validation data to plot")
    parser.add_argument("--random_assignment", type=bool, default=False, help="random assignment of clusters")
    parser.add_argument("--kl_weight", type=float, default=0.001, help="initial kl weight")
    parser.add_argument("--mmd_weight", type=float, default=0.1, help="initial mmd weight")
    parser.add_argument("--reconstruction_weight", type=float, default=1.0, help="initial reconstruction weight")
    parser.add_argument("--kl_anneal_warmup_epochs", type=int, default=20, help="initial warmup epochs")
    parser.add_argument("--kl_anneal_max", type=float, default=0.05, help="random assignment of clusters")
    #### clean up arguments #####
    parser.add_argument("--filtered_data_output", type=str, default="filtered_data.npz", help="cleaned up data")
    parser.add_argument("--conf_data", type=str, default="conf_data.npz", help="data with confidence scores")
    parser.add_argument("--plots", type=str, default="data", help="baseline plot name")
    parser.add_argument("--labels", type=str, default=None, help="label used for cleanup/leiden if None")
    parser.add_argument("--cluster_min", type=float, default=0.2, help="min confidence for per cluster cutoff")
    parser.add_argument("--quantil", type=float, default=0.2, help="confidence quantil")
    parser.add_argument("--quantil_max", type=float, default=0.3, help="max confidence for per cluster cutoff")
    parser.add_argument("--no_cleanup", action=argparse.BooleanOptionalAction, default = False)
    #### additional load fields #####
    parser.add_argument("--obs_fields",nargs="+",type=str,help="list of observation fields", default=[])
    parser.add_argument("--layer_fields",nargs="+",type=str, help="list of layer fields", default=[])
    
    args = parser.parse_args()
    set_seed()    
    ############################################################################################################
    print("#################################### running pre-processing #########################################")
    
    preprocessing(marker_gene_list = args.marker_genes, dataset = args.dataset_path, full_name = args.output_h5ad, batch_key=args.dataset_symbol, no_hvg = args.no_hvg)
    print(args.obs_fields)
    print(args.dataset_symbol)
    obs_fields = args.obs_fields
    obs_fields.append(args.dataset_symbol)
    obs_fields = list(set(obs_fields))
    layer_fields = args.layer_fields
    translate(args.output_h5ad, args.npy_dataset, obs_fields, layer_fields)
    
    
    ############################################################################################################
    print("######################################## training hvae ##############################################")
    
    gene_names = [line[:-1] for line in open(args.marker_genes).readlines()]
    
    main(train_data_name = args.npy_dataset, gene_names = gene_names, dataset_symbol = args.dataset_symbol,
          label_symbols = args.label_symbols,  data_symbol = args.data_symbol, gene_symbol = args.gene_symbol, 
            val_data_path = args.val_data_path, num_epochs = args.epochs, latent_dim = args.latent_dimension, hidden_dim = args.hidden_dimension, 
            layers = args.layers, learning_rate = args.learning_rate, batch_size = args.batch_size, patience = args.patience, 
            encode_covariates =args.encode_covariates, ln = args.ln, gene_factor = args.gene_factor, kl_weight = args.kl_weight, 
            mmd_weight = args.mmd_weight, reconstruction_weight = args.reconstruction_weight, save_embedding = args.output,
              kl_anneal_warmup_epochs = args.kl_anneal_warmup_epochs, kl_anneal_max = args.kl_anneal_max, model_checkpoint = args.model, 
              count_symbol = args.count_symbol, random_assignment = args.random_assignment)
    
    
    ############################################################################################################
    print("######################################### data cleanup ##############################################")
    
    if args.no_cleanup == False: clean_up(args.output, args.filtered_data_output, args.conf_data, plots=args.plots, labels = args.labels, cluster_min = args.cluster_min, quantil = args.quantil, quantil_max = args.quantil_max)
