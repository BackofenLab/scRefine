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


def main(train_data_name, gene_names, dataset_symbol, label_symbols,  data_symbol, gene_symbol, embedding_save, latent_dim = 1024, learning_rate = 0.0001, batch_size = 32, num_epochs = 100, eval_every = 1, print_every = 1, temperature = 0.5, sample_fraction = 0.1, contrastive_model = "contrastive_model.pkl"):

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = SingleCellDataset(gene_names = gene_names, dataset_symbol = dataset_symbol, label_symbols = label_symbols,  data_symbol = data_symbol, gene_symbol = gene_symbol, data_path=train_data_name, sample_fraction=sample_fraction, return_label_ints = True)

    #### use full for plotting
    val_dataset = SingleCellDataset(gene_names = gene_names, dataset_symbol = dataset_symbol, label_symbols = label_symbols,  data_symbol = data_symbol, gene_symbol = gene_symbol, data_path=train_data_name, sample_fraction=1.0, return_label_ints = True)
    
    input_size = train_dataset.data.shape[-1]
    model = Encoder(input_size, latent_dim=latent_dim).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
         val_dataset, batch_size=batch_size, shuffle=False
    )

    #-----------------------------------------------------------------------------

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(dataloader):
            data, labels, _, _ = batch_data
            data = data.to(device).float()
            optimizer.zero_grad()


            z = model(data)
            loss = nt_xent_loss(z, labels, temperature=temperature)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
            )

        if (epoch + 1) % eval_every == 0:
            torch.save(model.state_dict(), contrastive_model)


    model.eval()
    embeddings = []
    labels_train = []


    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):
            data, labels, _, _ = batch_data
            data = data.to(device).float()
            z = model(data)
            embeddings.extend(z.cpu().numpy())
            labels_train.extend(labels.numpy())

    
    
    np.savez(embedding_save, embeddings = embeddings, labels = labels_train)

def nt_xent_loss(z, labels, temperature=0.5):
    """
    z: latent dimension
    labels: labels of points in latent dimension
    temperature: parameter to scale the similarity matrix
    
    """
    
    batch_size = z.shape[0]

    z = torch.nn.functional.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T)
    similarity_matrix = similarity_matrix / temperature

    positive_mask = (
        (labels.unsqueeze(1) == labels.unsqueeze(0))
        .float()
        .to(similarity_matrix.device)
    )

    mask = torch.eye(batch_size, dtype=torch.bool).to(similarity_matrix.device)

    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    exp_logits = torch.exp(logits) * (~mask)
    exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

    log_prob = logits - torch.log(exp_logits_sum + 1e-10)

    positive_log_prob = (positive_mask * log_prob).sum(dim=1) / (
        positive_mask.sum(dim=1) + 1e-10
    )
    
    loss = -positive_log_prob.mean()

    return loss

if __name__ == "__main__":


    cmdline_parser = argparse.ArgumentParser('run contrastive learning')


    cmdline_parser.add_argument('-d', '--train_data_name',
                                help='dataset',
                                default = '/media/sven/Intenso/projects/backup/test_query.npz',
                                type=str)
    cmdline_parser.add_argument('-e', '--embedding_save',
                                help='embedding_save',
                                default = 'embedding_simclr.npz',
                                type=str)
    cmdline_parser.add_argument('-s', '--dataset_symbol',
                                help='dataset',
                                default = 'batch',
                                type=str)
    cmdline_parser.add_argument('-l', '--label_symbols',
                                help='symbol',
                                default = 'cell_subtype',
                                type=str)
    cmdline_parser.add_argument('-m', '--data_symbol',
                                help='data_symbol',
                                default = 'data',
                                type=str)
    cmdline_parser.add_argument('-g', '--gene_symbol',
                                help='gene_names',
                                default = 'gene_names',
                                type=str)
                                
    args, unknowns = cmdline_parser.parse_known_args()

    gene_names = ["PTPRC", "CD2", "CD3D","CD3G","CD3E", "CD8A","CD8B", "CD4", "MZB1", "CD19", "MS4A1", "CD14", "FCGR3A", "FCGR1A", "ITGAM", "CD68", "SPP1", "IL3RA",  "C1QA", "TYROBP", "FCER1G", "CPA3",  "TPSAB1", "MS4A2", "KIT", "ACTA2", "THY1", "CRABP1", "VWF", "ENG", "CDH5", "VCAM1", "KRT19", "KRT8", "EPCAM", "ALB", "TF", "TTR", "MLANA", "TYRP1", "MITF", "PMEL", "FDCSP", "SPIB", "ASCL1", "CALCA", "CHGA", "CHGB", "TFF3","GRP", "CEACAM5", "CEACAM6", "SCGB2A2","ENO2","SSTR2"]
    

    main(train_data_name = args.train_data_name, gene_names = gene_names, dataset_symbol = args.dataset_symbol, label_symbols = args.label_symbols,  data_symbol = args.data_symbol, gene_symbol = args.gene_symbol, embedding_save = args.embedding_save, latent_dim = 1024, learning_rate = 0.0001, batch_size = 32, num_epochs = 100, eval_every = 1, print_every = 500, temperature = 0.5)
