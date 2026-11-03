# scRefine

Batch-corrected latent representations of single-cell RNA-seq data with a hierarchical
VAE (HVAE), followed by confidence-based filtering of the resulting embedding.

The pipeline has three stages, all run by `main.py`:

1. **Preprocessing** (`preprocessing/create_data.py`): loads the marker-gene list and
   optionally selects HVGs / splits the data (both disabled in the standard run).
2. **Training** (`train_hvae.py`, `models/vae.py`): a ZINB-decoder VAE with
   - learnable GMM priors over the latent `z` and the per-batch library size, where each
     component's parameters are a shared base plus a per-component deviation
     (`--no-shared_base_prior` for independent per-component parameters),
   - a library-size KL selectable with `--library_kl`: `original` (default; sums every
     component's KL), `marginalized` (weights each component's KL by `q_pi`) or `batch`
     (only each cell's batch-label component, matching how the library is sampled),
   - optionally (`--decoder_covariates`) the batch embedding injected into the decoder,
   - an RBF-kernel MMD term between `q(z)` and the prior (batch integration),
   - a marker-gene contrastive head: an NT-Xent loss on a projection of `z` in which
     pairs of cells are weighted as positives by a Laplacian kernel over their
     log-normalized expression of each marker gene, averaged over marker genes,
   - a linearly annealed KL weight.

   Minibatches are built by a domain-aware sampler (`dataset/sampler.py`).
3. **Cleanup** (`clean_data.py`, `plotting.py`): Leiden clustering + t-SNE of the
   embedding, a radius-neighbors confidence model, and per-cluster filtering of
   low-confidence cells. Writes Plotly HTML plots.

## Installation

```bash
conda env create -f environment.yml
conda activate scRefine
```

`environment.yml` pins the minor versions scRefine was tested with (Python 3.11,
PyTorch 2.14, scanpy 1.11, PyTorch Lightning 2.6, Pyro 1.9) and installs the CUDA 12.6
build of PyTorch. A GPU is used automatically when available; to check that PyTorch sees
it:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False` on a GPU machine, your NVIDIA driver is likely too old for CUDA
12.6. Change `cu126` in `environment.yml` to a CUDA version your driver supports (see
https://pytorch.org/get-started).

## Input

An `.h5ad` file with raw counts in `.X`, and in `.obs`:

- a batch/domain column (default in `run.sh`: `dataset`),
- a cell-type label column (default in `run.sh`: `azimuth_final_level_label`).

The marker genes in `marker_genes.txt` (one symbol per line) should be present in
`var_names`. Replace the file to use a different marker panel.

## Running

```bash
./run.sh /path/to/data.h5ad          # seed 42
./run.sh /path/to/data.h5ad 1        # another seed
./run.sh /path/to/data.h5ad 42 --epochs 100 --no_plots   # extra main.py arguments
```

Edit `DATASET_SYMBOL` / `LABEL_SYMBOL` at the top of `run.sh` to match your data. Outputs
go to the current working directory. See `python main.py --help` for all options.

### Presets

`--preset` selects a named parameter set from `PRESETS` in `main.py`:

| Preset | Setting | Marker-gene head | Differs from the other preset in |
| --- | --- | --- | --- |
| `1` | contrastive | NT-Xent loss on a projection of `z` (`use_gene_contrastive=True`, temperature 0.5) | `encode_covariates=True`, `shared_base_prior=True` |
| `2` | classifier | linear head predicting binned marker-gene expression, MSE loss (`use_gene_contrastive=False`) | `encode_covariates=False`, `shared_base_prior=False` |

A preset only replaces the parser defaults; any argument passed explicitly on the command
line still takes precedence:

```bash
python main.py --preset 2 --epochs 100 ...   # preset 2, but trained for 100 epochs
```

Note that `run.sh` passes several of these arguments explicitly (including
`--use_gene_contrastive`), so they override a preset given as an extra argument. To run
preset `2` through `run.sh`, also pass `--no-use_gene_contrastive`, or call `main.py`
directly.

## Label-supervised alternatives (SimCLR / classification)

The scripts learn an embedding directly from existing
hard labels, such as cell-type annotations. Both read the same `.h5ad` input (raw counts
in `.X`).

### SimCLR-style contrastive embedding (`train_simclr.py`)

An MLP `Encoder` (`models/encoder.py`) is trained with a supervised NT-Xent loss: within
a minibatch, cells that share a label in `--label_symbol` are positives and all other
cells are negatives. Minibatches are plain shuffled batches with no domain-aware
sampling and no batch-correction term. Only a random `--sample_fraction` of the cells
(default 0.1) is used for training. All cells are embedded afterwards.

```bash
python train_simclr.py --data_path data.h5ad --dataset_symbol dataset \
  --label_symbol azimuth_final_level_label --epochs 100 \
  --embedding_save embedding_simclr.npz --plot_name simclr_tsne
```

Writes the embedding `.npz` (`embedding`, `labels`, `batches`, `obs_names`, same layout as
the HVAE output), the encoder weights (`--contrastive_model`), and optionally a t-SNE plot
(`--plot_name`). Other options: `--latent_dim` (default 1024), `--batch_size` (32),
`--temperature` (0.5).

### Classification / clustering on the embedding (`classify.py`)

`classify.py` first trains the SimCLR embedding above on `--data_path` (an `.h5ad`, with
the same SimCLR options). It saves the embedding to `--embedding_save` and then evaluates
it, depending on `--run_type`:

- `nn` (default): trains a small MLP classifier (`models/classifier.py`) on the embedding
  to predict the labels (90/10 train/validation split) and prints train/validation
  accuracy per epoch. Saves the classifier weights to `--model_path` and the
  validation labels, predictions and raw outputs to `--evaluation` (written with
  `np.savez`, so give it an `.npz` name).
- `kmeans`: runs K-Means for several cluster counts and maps each cluster to the
  majority label of 50 sampled cells. It then fits a k-NN classifier on these pseudo-labels
  and prints accuracy, weighted F1 and silhouette scores against the true labels.
- `predict`: skips training. It loads an existing embedding (`--embedding_save`) and
  classifier (`--model_path`) and saves the predicted label indices to `--evaluation`
  (`.npy`).

```bash
python classify.py --data_path data.h5ad --dataset_symbol dataset \
  --label_symbol azimuth_final_level_label --epochs 100 --run_type nn \
  --embedding_save embedding_simclr.npz --model_path classifier.pth \
  --evaluation classification.npz
```

Because both approaches are supervised by the labels, they only work on data that is
already annotated. 

## Outputs

With prefix `screfine_seed<SEED>`:

| File | Content |
| --- | --- |
| `<prefix>.npz` | HVAE embedding: `embedding`, `batches`, `labels`, `obs_names` |
| `<prefix>.ckpt` | trained model checkpoint |
| `<prefix>_conf.npz` | embedding with per-cell confidence and t-SNE coordinates |
| `<prefix>_filtered.npz` | cells kept after confidence filtering |
| `plots_<prefix>_*.html` | t-SNE plots (skipped with `--no_plots`) |
