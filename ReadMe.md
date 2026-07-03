scRefine

**Motivation** Variational autoencoders (VAEs) have become useful tools for learning meaningful latent representations of high-dimensional biological data, making them particularly useful for analyzing single-cell transcriptomic profiles. In the context of immune cell data, scVAEs can capture complex cellular heterogeneity, enabling the discovery of rare immune subpopulations and the identification of subtle transcriptional shifts during immune responses. 
Given the highly dynamic nature of immune cells, which transition between states in response to pathogens, inflammation, or disease, traditional clustering methods often struggle to capture continuous trajectories and subtle variations. By leveraging scVAEs, we can learn low-dimensional representations that preserve biologically relevant structures, facilitating better characterization of immune cell states and their functional roles. Additionally we introduce a novel approach to integrating data from different batches. 

**installing scRefine**

#### Miniconda
Make sure to install miniconda first. Then create an env with: 

```
conda create --name scRefine --file environment.yml
```

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

Choose --run_type as nn to train a basic feedforward NN or kmeans to evaluate a kmeans classifier.  
