import hashlib
import numpy as np
import random
import torch
from models.vae import HVAE_Lightning
import os
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataset.dataloader import build_dataloaders
import scanpy as sc


class ResumeStateCallback(Callback):
    """
    Adds what Lightning leaves out of its checkpoints but an exact resume needs: the RNG
    states, and the settings used to refuse resuming with a different configuration.
    """

    def __init__(self, run_config):
        """
        Args:
            run_config (dict): Settings and data fingerprints not covered by the model's
                hyperparameters; see `_check_resume_compatible`.
        """
        self.run_config = run_config
        self._rng_state = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Stores the run config and the Python/NumPy/PyTorch RNG states in `checkpoint`."""
        checkpoint["run_config"] = self.run_config
        checkpoint["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Keeps the saved RNG states until `on_fit_start`."""
        self._rng_state = checkpoint.get("rng_state")

    def on_fit_start(self, trainer, pl_module):
        """
        Restores the RNG states. This has to happen here: the fit loop creates the data
        iterator, which draws the first minibatches, before `on_train_start`.
        """
        if self._rng_state is None:
            return
        random.setstate(self._rng_state["python"])
        np.random.set_state(self._rng_state["numpy"])
        torch.set_rng_state(self._rng_state["torch"].cpu())
        if torch.cuda.is_available():
            for device_id, state in enumerate(self._rng_state["cuda"][:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state.cpu(), device_id)
        self._rng_state = None


def interim_checkpoint_paths(model_checkpoint):
    """
    Returns `(directory, file)` of the checkpoint written during training for the final
    checkpoint path `model_checkpoint`.
    """
    directory = os.path.splitext(model_checkpoint)[0] + "_ckpts"
    return directory, os.path.join(directory, "last.ckpt")


def check_no_incomplete_run(model_checkpoint, resume):
    """
    Raises if an aborted training run left its checkpoint behind and `resume` is False, so
    that starting over cannot silently overwrite its progress.

    Args:
        model_checkpoint (str): Final checkpoint path of the run.
        resume (bool): Whether the run was asked to resume.
    """
    directory, interim_checkpoint = interim_checkpoint_paths(model_checkpoint)
    if not resume and os.path.exists(interim_checkpoint):
        raise FileExistsError(
            f"Found the checkpoint of an incomplete training run: {interim_checkpoint}. "
            f"Pass --resume to continue it, or delete {directory} to start over."
        )


def _fingerprint(values):
    """Returns a hash of a sequence of values, to detect changed input data."""
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _check_resume_compatible(checkpoint, checkpoint_path, model, run_config):
    """
    Raises if `checkpoint` was written with other model hyperparameters, training settings
    or input data than the current run.

    Args:
        checkpoint (dict): Loaded checkpoint.
        checkpoint_path (str): Path of the checkpoint, for the error message.
        model (HVAE_Lightning): Model of the current run.
        run_config (dict): Settings and data fingerprints of the current run.
    """
    saved = {**checkpoint["hyper_parameters"], **checkpoint.get("run_config", {})}
    current = {**model.hparams, **run_config}
    differences = [f"{key}: checkpoint={saved[key]!r}, current={current[key]!r}"
                   for key in sorted(saved.keys() & current.keys()) if saved[key] != current[key]]
    if differences:
        raise ValueError(
            f"Cannot resume from {checkpoint_path}, the run differs from the checkpoint in:\n  "
            + "\n  ".join(differences)
        )


def save_validate(dataloader, model, save_embedding, obs_names=None):
    """
    Runs `model` over `dataloader` in eval mode and saves the resulting latent means,
    domain/batch labels, and input labels to an `.npz` file.

    Args:
        dataloader (DataLoader): Yields `(data, label_data, batched_data, gene)` batches.
        model (HVAE_Lightning): Trained model providing `forward` and returning `z_mu`.
        save_embedding (str): Output `.npz` path for the embeddings/batches/labels.
        obs_names (array-like, optional): Cell barcodes, in the exact order `dataloader`
            yields them (true for the embedding loader from `build_dataloaders`, which
            iterates over all cells in their original order). Saved alongside the embedding so a
            downstream evaluation step can join back to the source AnnData by identity
            instead of assuming row order. Raises if its length doesn't match the number
            of embedded cells.
    """
    device = "cpu"

    model.eval()
    # The Trainer runs at 64-bit precision, so the model's dtype differs from the collator's.
    model_dtype = next(model.parameters()).dtype

    num_cells = len(dataloader.dataset)
    mu_array = None
    batched_data_list = []
    label_data_list = []
    cursor = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            data, label_data, batched_data, gene = batch_data
            data = data.to(device, dtype=model_dtype)

            batched_data_list.extend(np.array(batched_data))
            if label_data is not None:
                label_data_list.extend(np.array(label_data))
            batched_data = batched_data.to(device)
            output = model(data, gene, batched_data)
            mu = output["z_mu"].detach().cpu().numpy()

            if mu_array is None:
                mu_array = np.empty((num_cells, mu.shape[1]), dtype=mu.dtype)
            mu_array[cursor:cursor + mu.shape[0]] = mu
            cursor += mu.shape[0]

    mu_array = mu_array[:cursor]
    save_kwargs = dict(embedding = mu_array, batches = np.array(batched_data_list),
                        labels = np.array(label_data_list))

    if obs_names is not None:
        obs_names = np.asarray(obs_names)
        if len(obs_names) != len(mu_array):
            raise ValueError(
                f"obs_names has {len(obs_names)} entries but {len(mu_array)} cells were "
                "embedded; the dataloader must iterate over the full, unshuffled "
                "dataset for obs_names order to match the embedding order."
            )
        save_kwargs["obs_names"] = obs_names

    np.savez(save_embedding, **save_kwargs)

    return


        

        
def main(train_data_name, gene_names, dataset_symbol, label_symbols, 
         data_symbol, gene_symbol, val_data_path = None, 
         num_epochs = 2, latent_dim = 90, hidden_dim = 128, 
         layers = 1, learning_rate = 0.001, batch_size = 33,
           patience = 15, encode_covariates =True, ln = False, 
           gene_factor = 20, kl_weight = 0.01, 
            mmd_weight = 0.5, reconstruction_weight = 1.0, 
            kl_anneal_warmup_epochs = 20, kl_anneal_max = 0.5, 
             save_embedding = "embedding.npz",
             model_checkpoint = "model.ckpt", count_symbol = None, random_assignment = False,
             use_gene_contrastive = False, gene_contrastive_temperature = 0.5,
             gene_contrastive_bandwidth = None, gene_contrastive_cross_batch_only = True,
             library_kl = "original", shared_base_prior = True, decoder_covariates = False,
             resume = False, checkpoint_every_n_epochs = 1,
            enable_progress_bar = True):
    """
    Trains an `HVAE_Lightning` model (whose GMM priors optionally parameterize each
    component's mean/(log-variance or log-std) as a shared base plus a per-component
    deviation, see `shared_base_prior`) on
    single-cell data: builds domain-aware DataLoaders, fits the model with early stopping
    on training loss, saves the model checkpoint, and embeds all cells.

    Args:
        train_data_name (str): Path to the input `.h5ad` training dataset.
        gene_names (list): Marker gene symbols used for the auxiliary gene-prediction head.
        dataset_symbol (str): Column in `adata.obs` used as the domain/batch key.
        label_symbols (str): Unused by this function directly; kept for CLI/pipeline parity.
        data_symbol (str): Unused by this function directly; kept for CLI/pipeline parity.
        gene_symbol (str): Unused by this function directly; kept for CLI/pipeline parity.
        val_data_path (str, optional): Unused by this function; reserved for future support
            of a separate validation dataset.
        num_epochs (int, optional): Maximum number of training epochs.
        latent_dim (int, optional): Dimensionality of the VAE latent space.
        hidden_dim (int, optional): Hidden layer width for the encoder/decoder.
        layers (int, optional): Number of hidden layers in the encoder/decoder.
        learning_rate (float, optional): Initial learning rate of the model's Adam optimizer.
        batch_size (int, optional): Minibatch size for training and for embedding the data.
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
        save_embedding (str, optional): Output `.npz` path for the embeddings.
        model_checkpoint (str, optional): Output path for the Lightning model checkpoint.
        count_symbol (str, optional): Unused by this function; reserved for a normalized
            count layer/column selector.
        random_assignment (bool, optional): Whether the model samples a random GMM component
            for the library size instead of using the observed batch label.
        use_gene_contrastive (bool, optional): If True, replace the MSE-based marker-gene
            prediction head with NT-Xent contrastive learning directly on raw (unbinned)
            log-normalized marker-gene expression; also disables binning in the
            `AnnDataCollator` for the marker-gene targets.
        gene_contrastive_temperature (float, optional): NT-Xent softmax temperature, only
            used when `use_gene_contrastive=True`.
        gene_contrastive_bandwidth (float, optional): Kernel bandwidth for the
            expression-similarity weighting; only used when `use_gene_contrastive=True`
            (default: per-gene batch std).
        gene_contrastive_cross_batch_only (bool, optional): Only used when
            `use_gene_contrastive=True`. If True (default), the contrastive loss only
            compares cross-batch pairs (batch-integration signal). If False, all pairs are
            compared by marker-gene-expression similarity regardless of batch, favoring
            cell-type structure over batch alignment.
        library_kl (str, optional): Library-size KL variant passed to `HVAE_Lightning`:
            `"original"`, `"marginalized"` or `"batch"`.
        shared_base_prior (bool, optional): Parameterize the GMM priors as a shared base
            plus per-component deviations (True) or independently per component (False).
        decoder_covariates (bool, optional): Inject the batch covariate embedding into the
            decoder, so batch effects can be explained by the decoder instead of `z`.
        enable_progress_bar (bool, optional): Show the Lightning per-step progress bar.
        resume (bool, optional): Continue from the checkpoint of an aborted run. If training
            had already finished (only `model_checkpoint` exists), the model is loaded and
            only the embedding is redone. Starts from scratch if neither checkpoint exists.
        checkpoint_every_n_epochs (int, optional): How often to overwrite the checkpoint
            `<model_checkpoint without extension>_ckpts/last.ckpt` during training. It is
            removed once training finishes.

    Returns:
        None
    """
    if checkpoint_every_n_epochs < 1:
        raise ValueError(f"checkpoint_every_n_epochs must be at least 1, got {checkpoint_every_n_epochs}")
    check_no_incomplete_run(model_checkpoint, resume)
    checkpoint_dir, interim_checkpoint = interim_checkpoint_paths(model_checkpoint)
    resume_from = None
    if resume:
        # The interim checkpoint comes first: if both exist, it belongs to a later run than
        # the final checkpoint.
        if os.path.exists(interim_checkpoint):
            resume_from = interim_checkpoint
        elif os.path.exists(model_checkpoint):
            resume_from = model_checkpoint
        else:
            print(f"No checkpoint found for {model_checkpoint}, starting from scratch.")
    training_finished = resume_from == model_checkpoint

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    train_adata = sc.read(os.path.join(train_data_name))

    train_dataloader, embed_dataloader, info = build_dataloaders(
            train_adata, dataset_symbol, gene_names,
            batch_size=batch_size,
            p_intra_domain=0.7,
            bin_marker_genes=not use_gene_contrastive,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    )

    model = HVAE_Lightning(n_input=info.n_genes, n_latent=latent_dim,n_hidden = hidden_dim, n_layers=layers,
                            lr = learning_rate,
                            encode_covariates=encode_covariates, ln = ln, num_components = info.n_batches, 
                            gene_factor = gene_factor, prediction_genes = info.n_marker_genes, 
                            n_batch = info.n_batches, random_assignment = random_assignment, kl_weight = kl_weight,
                            mmd_weight = mmd_weight, reconstruction_weight = reconstruction_weight,
                            kl_anneal_warmup_epochs = kl_anneal_warmup_epochs, kl_anneal_max = kl_anneal_max,
                            use_gene_contrastive = use_gene_contrastive,
                            gene_contrastive_temperature = gene_contrastive_temperature,
                            gene_contrastive_bandwidth = gene_contrastive_bandwidth,
                            gene_contrastive_cross_batch_only = gene_contrastive_cross_batch_only,
                            library_kl = library_kl, shared_base_prior = shared_base_prior,
                            decoder_covariates = decoder_covariates)
    model.to(device)
    model.get_dataset_baseline(info.lib_sizes, info.batch_ids, n_batch = info.n_batches)

    # `patience` is listed because EarlyStopping restores it from the checkpoint, which would
    # silently override a changed value.
    run_config = dict(
        batch_size=batch_size,
        patience=patience,
        cells_and_batches=_fingerprint([*train_adata.obs_names, *train_adata.obs[dataset_symbol]]),
        genes=_fingerprint(train_adata.var_names),
        marker_genes=_fingerprint(gene_names),
    )

    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        _check_resume_compatible(checkpoint, resume_from, model, run_config)
        # Lightning loads a checkpoint before applying `precision=64`, which would otherwise
        # round the saved weights to 32 bits on their way into the model.
        model.double()

    early_stop_callback = EarlyStopping(
    monitor="train_loss",  
    patience=patience,          
    mode="min",            
    verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="last",
    every_n_epochs=checkpoint_every_n_epochs,
    save_on_train_epoch_end=True,
    enable_version_counter=False,
    )

    if training_finished:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        trainer = Trainer(
        max_epochs=num_epochs, 
        callbacks=[early_stop_callback, ResumeStateCallback(run_config), checkpoint_callback],
        precision = 64,
        enable_progress_bar = enable_progress_bar,
        )
        if resume_from is not None:
            # The sampler draws random numbers the first time its length is requested, which
            # an uninterrupted run did long before this point. Do that before the RNG states
            # are restored.
            len(train_dataloader.batch_sampler)
        # weights_only=False: the checkpoint holds RNG states and is written by this function.
        trainer.fit(model, train_dataloader, ckpt_path=resume_from, weights_only=False)
        trainer.save_checkpoint(model_checkpoint)
        if os.path.exists(interim_checkpoint):
            os.remove(interim_checkpoint)
            if not os.listdir(checkpoint_dir):
                os.rmdir(checkpoint_dir)

    model.to("cpu")
    # embed_dataloader iterates over all cells in `train_adata`'s original order.
    save_validate(embed_dataloader, model, save_embedding, obs_names=train_adata.obs_names.to_numpy())
