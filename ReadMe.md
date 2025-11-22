scRefine

**Motivation** Variational autoencoders (VAEs) have become useful tools for learning meaningful latent representations of high-dimensional biological data, making them particularly useful for analyzing single-cell transcriptomic profiles. In the context of immune cell data, scVAEs can capture complex cellular heterogeneity, enabling the discovery of rare immune subpopulations and the identification of subtle transcriptional shifts during immune responses. 
Given the highly dynamic nature of immune cells, which transition between states in response to pathogens, inflammation, or disease, traditional clustering methods often struggle to capture continuous trajectories and subtle variations. By leveraging scVAEs, we can learn low-dimensional representations that preserve biologically relevant structures, facilitating better characterization of immune cell states and their functional roles. Additionally we introduce a novel approach to integrating data from different batches. 


**running scRefine**

#### Example
```
python main.py \
--dataset_path {dataset} \
--dataset_symbol {field for batch correction} \
--no_hvg \
--label_symbols {labels for plotting} \
--output {output embedding} \
--filtered_data_output {filtered output} \
--conf_data {output with conf data} \
--plots {base name for potting} 
```
**contrastive learning**

#### Example

before running the contrastive learning script, please use preprocess.py to create the data.

```
python preprocess.py --dataset {dataset name} --dataset_key {batch name} --obs_fields {obs fields to be copied} --layer_fields {layer fields to be copied} --npy_dataset {npz output}
```

then use train_simclr.py


```
python train_simclr.py --train_data_name {data file} --embedding_save {embedding-output} --dataset_symbol {name of dataset field} --label_symbols {name of label field} --data_symbol {name of data field}
```

