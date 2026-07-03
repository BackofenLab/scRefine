import numpy as np
import torch
from models.encoder import Encoder
import scanpy as sc
import os
from sklearn.manifold import TSNE
import argparse
from dataset.dataloader import DataLoaderManager
from plotting import plot_tsne


def main(train_data_name,
         gene_names,
         dataset_symbol,
         label_symbols,
         embedding_save,
         latent_dim = 1024,
         learning_rate = 0.0001,
         batch_size = 32,
         num_epochs = 100,
         eval_every = 1,
         print_every = 1,
         temperature = 0.5,
         sample_fraction = 0.1,
         contrastive_model = "contrastive_model.pkl",
         plot_name = None):
    """
    Trains a SimClr-style contrastive `Encoder` on single-cell data with an NT-Xent loss,
    then embeds the validation split and saves the resulting embeddings/labels.

    Args:
        train_data_name (str): Path to the input `.h5ad` dataset.
        gene_names (list): Marker gene symbols forwarded to the data loader (may be empty).
        dataset_symbol (str): Column in `adata.obs` used as the domain/batch key.
        label_symbols (str): Column in `adata.obs` used as the class key for the
            contrastive loss's positive-pair mask.
        embedding_save (str): Output `.npz` path for the saved embeddings and labels.
        latent_dim (int, optional): Dimensionality of the encoder's projection head output.
        learning_rate (float, optional): Adam learning rate.
        batch_size (int, optional): Training batch size.
        num_epochs (int, optional): Number of training epochs.
        eval_every (int, optional): Epoch interval at which the model checkpoint is saved.
        print_every (int, optional): Epoch interval at which training loss is printed.
        temperature (float, optional): Temperature for the NT-Xent contrastive loss.
        sample_fraction (float, optional): Fraction of data used for training (passed as
            `train_frac` to `DataLoaderManager`); the remainder is used for validation.
        contrastive_model (str, optional): Output path for the trained encoder's state dict.
        plot_name (str, optional): If given, saves a TSNE plot of the final embeddings here.
    """
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)


    train_adata = sc.read(os.path.join(train_data_name))

    data_manager = DataLoaderManager(
            domain_key=dataset_symbol,
            class_key=label_symbols,
            covariate_keys=None,
            feature_list=None,
            normalize_total=False,
            log1p=False,
            batch_size=batch_size,
            train_frac=sample_fraction,
            use_sampler=True,
            sampler_emb=None,
            sampler_knn=None,
            sampler_domain_minibatch_strategy='proportional',
            domain_coverage=None,
            dist_metric='euclidean',
            p_intra_knn=0.0,
            p_intra_domain=0.0,
            use_faiss=True,
            use_ivf=True,
            ivf_nprobe=10,
            preload_dense=False,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )

    train_dataloader, val_dataloader, data_structure, info = data_manager.anndata_to_dataloader(train_adata, dataset_symbol,gene_names)
    batches_out,data_out,input_size, num_batches, number_marker_genes = info

    model = Encoder(input_size, latent_dim=latent_dim).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #-----------------------------------------------------------------------------

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(train_dataloader):
            data, labels, _, _ = batch_data
            data = data.to(device).float()
            optimizer.zero_grad()


            z = model(data)
            loss = nt_xent_loss(z, labels, temperature=temperature)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
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

    
    
    np.savez(embedding_save, embedding = embeddings, labels = labels_train)

    if plot_name != None:
        tsne = TSNE(n_components=2, random_state=0)
        tsne_projection = tsne.fit_transform(np.array(embeddings))
        plot_tsne(tsne_projection, labels_train, plot_name)



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


    cmdline_parser.add_argument('-d', '--data_path',
                                help='dataset',
                                default = 'input.h5ad',
                                type=str)
    cmdline_parser.add_argument('-e', '--embedding_save',
                                help='embedding_save',
                                default = 'embedding_simclr.npz',
                                type=str)
    cmdline_parser.add_argument('-s', '--dataset_symbol',
                                help='dataset',
                                default = 'batch',
                                type=str)
    cmdline_parser.add_argument('-l', '--label_symbol',
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
    cmdline_parser.add_argument('-z', '--latent_dim',
                                help='latent dim of contrastive model',
                                default = 1024,
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                help='batch size',
                                default = 32,
                                type=int)
    cmdline_parser.add_argument('-n', '--epochs',
                                help='epochs to train',
                                default = 100,
                                type=int)
    cmdline_parser.add_argument('-t', '--temperature',
                                help='model temp',
                                default = 0.5,
                                type=float)
    cmdline_parser.add_argument('-c', '--contrastive_model',
                                help='contrastive_model',
                                default = "contrastive_model.pkl",
                                type=str)
    cmdline_parser.add_argument('-p', '--plot_name',
                                help='plot name',
                                default = None,
                                type=str)

    args, unknowns = cmdline_parser.parse_known_args()

    main(train_data_name=args.data_path, gene_names=[],
            dataset_symbol=args.dataset_symbol, label_symbols=args.label_symbol,
            embedding_save=args.embedding_save, latent_dim=args.latent_dim, learning_rate=0.0001,
            batch_size=args.batch_size, num_epochs=args.epochs,
            eval_every=1, print_every=1,
            temperature=args.temperature, contrastive_model = args.contrastive_model, plot_name = args.plot_name)
