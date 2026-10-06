# scRefine

**Motivation** Variational autoencoders (VAEs) have become useful tools for learning meaningful latent representations of high-dimensional biological data, making them particularly useful for analyzing single-cell transcriptomic profiles. In the context of immune cell data, scVAEs can capture complex cellular heterogeneity, enabling the discovery of rare immune subpopulations and the identification of subtle transcriptional shifts during immune responses. 
Given the highly dynamic nature of immune cells, which transition between states in response to pathogens, inflammation, or disease, traditional clustering methods often struggle to capture continuous trajectories and subtle variations. By leveraging scVAEs, we can learn low-dimensional representations that preserve biologically relevant structures, facilitating better characterization of immune cell states and their functional roles. Additionally we introduce a novel approach to integrating data from different batches. 

**installing scRefine**

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



#### Docker
tbd




**running scRefine**

#### Example
```
python main.py \
--dataset_path {path_to_input_adata} \
--dataset_symbol {symbol for dataset integration} \
--no_hvg \
--no-split_data \
--label_symbols {symbol for labels to be saved} \
--output {output embedding} \
--filtered_data_output "{filtered output embedding} \
--conf_data {output embedding with prediction confidence} \
--plots {base name for potting}  \
--marker_genes {marker gene text file} \
--epochs {max number of training epochs} 
```

run with *--no_hvg* and *--no_split_data* to not use hvg-selection and not split the data into training/testing partitions. 

## Outputs

With prefix `screfine_seed<SEED>`:

| File | Content |
| --- | --- |
| `<prefix>.npz` | HVAE embedding: `embedding`, `batches`, `labels`, `obs_names` |
| `<prefix>.ckpt` | trained model checkpoint |
| `<prefix>_conf.npz` | embedding with per-cell confidence and t-SNE coordinates |
| `<prefix>_filtered.npz` | cells kept after confidence filtering |
| `plots_<prefix>_*.html` | t-SNE plots (skipped with `--no_plots`) |


**contrastive learning**

use train_simclr.py to create a contrastive learning model.

#### Example

```
python train_simclr.py \
--data_path {input h5ad} \
--dataset_symbol {symbol for the dataset} \
--label_symbol {symbol to be used for the contrastive training} \
--plot_name {plot_name if needed} \
--epochs {number of epochs to train} \

```

**classification**

#### Example

```
python classify.py \
--data_path {path_to_data}\
--dataset_symbol {batch data to save} \
--label_symbol {label data used for training} \
--epochs {number of epochs to train} \
```


