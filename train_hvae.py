import numpy as np
import torch
from models.vae import HVAE_Lightning
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from dataset.dataloader import DataLoaderManager
import scanpy as sc


def save_validate(dataloader, model, save_embedding):
    """
    Runs `model` over `dataloader` in eval mode and saves the resulting latent means,
    domain/batch labels, and input labels to an `.npz` file.

    Args:
        dataloader (DataLoader): Yields `(data, label_data, batched_data, gene)` batches.
        model (HVAE_Lightning): Trained model providing `forward` and returning `z_mu`.
        save_embedding (str): Output `.npz` path for the embeddings/batches/labels.
    """
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
            batched_data = batched_data.to(device)
            output = model(data, gene, batched_data)
            data_list.extend(data.to("cpu"))
            mu = output["z_mu"]
            mu_list.extend(mu.detach().cpu().numpy())
        
    np.savez(save_embedding, embedding = mu_list, batches = np.array(batched_data_list))

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
             model_checkpoint = "model.ckpt", random_assignment = False):
    """
    Trains an `HVAE_Lightning` model on single-cell data: builds domain-aware DataLoaders,
    fits the model with early stopping on training loss, saves the model checkpoint, and
    embeds the validation split.

    Args:
        train_data_name (str): Path to the input `.h5ad` training dataset.
        gene_names (list): Marker gene symbols used for the auxiliary gene-prediction head.
        dataset_symbol (str): Column in `adata.obs` used as the domain/batch key.
        label_symbols (str): Unused by this function directly; kept for CLI/pipeline parity.
        data_symbol (str): Key in `adata.layers` for the expression matrix to train on. If
            the key is present in `train_adata.layers`, it replaces `adata.X` before data
            loading; otherwise `adata.X` is used unchanged.
        gene_symbol (str): Unused by this function directly; kept for CLI/pipeline parity.
        val_data_path (str, optional): Unused by this function; reserved for future support
            of a separate validation dataset.
        num_epochs (int, optional): Maximum number of training epochs.
        latent_dim (int, optional): Dimensionality of the VAE latent space.
        hidden_dim (int, optional): Hidden layer width for the encoder/decoder.
        layers (int, optional): Number of hidden layers in the encoder/decoder.
        learning_rate (float, optional): Learning rate (an unused optimizer is also built
            with this value; the model's own optimizer, from `configure_optimizers`, is what
            the Lightning `Trainer` actually uses).
        batch_size (int, optional): Batch size (only used for the `AnnDataCollator`'s
            baseline library-size computation; the sampler determines actual batch sizes).
        patience (int, optional): Early-stopping patience on `train_loss`.
        encode_covariates (bool, optional): Whether to concatenate batch covariate
            embeddings into the encoder input.
        ln (bool, optional): Whether to use layer normalization in the decoder.
        gene_factor (int, optional): Weight of the auxiliary marker-gene prediction loss.
        kl_weight (float, optional): Starting KL weight for annealing.
        mmd_weight (float, optional): Weight of the MMD term in the latent KL loss.
        reconstruction_weight (float, optional): Weight of the reconstruction loss.
        kl_anneal_warmup_epochs (int, optional): Epochs over which KL weight is annealed.
        kl_anneal_max (float, optional): KL weight after annealing completes.
        save_embedding (str, optional): Output `.npz` path for the validation embeddings.
        model_checkpoint (str, optional): Output path for the Lightning model checkpoint.
        random_assignment (bool, optional): Whether the model samples a random GMM component
            for the library size instead of using the observed batch label.

    Returns:
        None
    """
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    train_adata = sc.read(os.path.join(train_data_name))

    if data_symbol is not None and data_symbol in train_adata.layers:
        train_adata.X = train_adata.layers[data_symbol]

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
