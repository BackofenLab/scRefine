import numpy as np
import torch
from models.encoder import Encoder
import scanpy as sc
import os
from sklearn.manifold import TSNE
import argparse
from torch.utils.data import DataLoader, Subset
from dataset.dataloader import AnnDataCollator, IndexDataset
from plotting import plot_tsne


def build_simclr_dataloaders(adata, dataset_symbol, label_symbols, batch_size,
                             sample_fraction, seed=42, num_workers=None):
    """
    Builds the DataLoaders for supervised contrastive training: a training loader over a
    random `sample_fraction` of the cells, in plain shuffled minibatches (no domain-aware
    sampling), and an embedding loader over all cells in their original order.

    Args:
        adata (AnnData): Input data with raw counts in `.X`.
        dataset_symbol (str): Column in `adata.obs` holding the batch/domain label.
        label_symbols (str): Column in `adata.obs` holding the class label used for the
            contrastive loss's positive-pair mask.
        batch_size (int): Minibatch size for both loaders.
        sample_fraction (float): Fraction of cells used for training.
        seed (int, optional): Seed for drawing the training cells.
        num_workers (int, optional): Training-loader workers; defaults to
            `min(4, os.cpu_count())`.

    Returns:
        tuple: `(train_loader, embed_loader)`. Both yield
            `(X, class_ids, batch_ids, None)` batches.
    """
    batch_labels = adata.obs[dataset_symbol].to_numpy()
    _, batch_ids = np.unique(batch_labels, return_inverse=True)
    _, class_ids = np.unique(adata.obs[label_symbols].astype(str).to_numpy(), return_inverse=True)

    collator = AnnDataCollator(adata, batch_ids, batch_labels, marker_genes=None, class_ids=class_ids)
    dataset = IndexDataset(adata.n_obs)

    n_train = int(sample_fraction * adata.n_obs)
    if n_train < 2:
        raise ValueError(
            f"sample_fraction={sample_fraction} leaves {n_train} training cells; at least 2 are needed."
        )
    train_indices = np.random.default_rng(seed).permutation(adata.n_obs)[:n_train]

    if num_workers is None:
        num_workers = min(4, os.cpu_count())

    # drop_last avoids a trailing minibatch of one cell, which BatchNorm cannot train on.
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True,
                              drop_last=n_train > batch_size, num_workers=num_workers,
                              collate_fn=collator, pin_memory=True)
    embed_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return train_loader, embed_loader


def main(train_data_name,
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
    Trains a SimClr-style contrastive `Encoder` on single-cell data with a supervised
    NT-Xent loss (cells sharing a label are positives) on a random `sample_fraction` of the
    cells, then embeds all cells and saves the resulting embeddings/labels.

    Args:
        train_data_name (str): Path to the input `.h5ad` dataset (raw counts in `.X`).
        dataset_symbol (str): Column in `adata.obs` used as the domain/batch key.
        label_symbols (str): Column in `adata.obs` used as the class key for the
            contrastive loss's positive-pair mask.
        embedding_save (str): Output `.npz` path for the embeddings (`embedding`,
            `labels`, `batches`, `obs_names`), in the order of the input `.h5ad`.
        latent_dim (int, optional): Dimensionality of the encoder's projection head output.
        learning_rate (float, optional): Adam learning rate.
        batch_size (int, optional): Training batch size.
        num_epochs (int, optional): Number of training epochs.
        eval_every (int, optional): Epoch interval at which the model checkpoint is saved.
        print_every (int, optional): Epoch interval at which training loss is printed.
        temperature (float, optional): Temperature for the NT-Xent contrastive loss.
        sample_fraction (float, optional): Fraction of cells used to train the encoder;
            all cells are embedded afterwards.
        contrastive_model (str, optional): Output path for the trained encoder's state dict.
        plot_name (str, optional): If given, saves a TSNE plot of the final embeddings here.
    """
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)


    train_adata = sc.read(os.path.join(train_data_name))

    train_dataloader, embed_dataloader = build_simclr_dataloaders(
            train_adata, dataset_symbol, label_symbols,
            batch_size=batch_size, sample_fraction=sample_fraction, seed=seed,
    )

    model = Encoder(train_adata.n_vars, latent_dim=latent_dim).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #-----------------------------------------------------------------------------

    # Training loop
    for epoch in range(num_epochs):
        model.train()
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


    with torch.no_grad():
        for batch_idx, batch_data in enumerate(embed_dataloader):
            data, _, _, _ = batch_data
            data = data.to(device).float()
            z = model(data)
            embeddings.extend(z.cpu().numpy())

    embeddings = np.array(embeddings)
    labels = train_adata.obs[label_symbols].astype(str).to_numpy()

    np.savez(embedding_save, embedding = embeddings, labels = labels,
             batches = train_adata.obs[dataset_symbol].astype(str).to_numpy(),
             obs_names = train_adata.obs_names.to_numpy())

    if plot_name != None:
        tsne = TSNE(n_components=2, random_state=0)
        tsne_projection = tsne.fit_transform(embeddings)
        plot_tsne(tsne_projection, labels, plot_name)



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
    cmdline_parser.add_argument('-f', '--sample_fraction',
                                help='fraction of cells used to train the encoder (all cells are embedded)',
                                default = 0.1,
                                type=float)
    cmdline_parser.add_argument('-c', '--contrastive_model',
                                help='contrastive_model',
                                default = "contrastive_model.pkl",
                                type=str)
    cmdline_parser.add_argument('-p', '--plot_name',
                                help='plot name',
                                default = None,
                                type=str)

    args = cmdline_parser.parse_args()

    main(train_data_name=args.data_path,
            dataset_symbol=args.dataset_symbol, label_symbols=args.label_symbol,
            embedding_save=args.embedding_save, latent_dim=args.latent_dim, learning_rate=0.0001,
            batch_size=args.batch_size, num_epochs=args.epochs,
            eval_every=1, print_every=1,
            temperature=args.temperature, sample_fraction=args.sample_fraction,
            contrastive_model = args.contrastive_model, plot_name = args.plot_name)
