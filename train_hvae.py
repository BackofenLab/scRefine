import numpy as np
import torch
from models.vae import  HVAE_Lightning
from dataset.singlecell import SingleCellDataset2
import logging
import time
import os
import json
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from pytorch_lightning import Trainer



def compute_batch_mixing_entropy(embedding, batch_labels, n_neighbors=50):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    entropies = []
    
    for i in range(len(indices)):

        neighbor_batches = batch_labels[indices[i]]
        unique, counts = np.unique(neighbor_batches, return_counts=True)
        probs = counts / len(neighbor_batches)
        entropies.append(entropy(probs))
    
    average_entropy = np.mean(entropies)
    
    return average_entropy
    
    
def save_validate(dataloader, epoch, model):
    cuda_id = 0
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    mu_list = []
    batched_data_list = []
    label_data_list = []
    data_list = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            data, label_data,_, batched_data = batch_data
            data = data.to(device).float()
            batched_data_list.extend(np.array(batched_data))
            label_data_list.extend(np.array(label_data))
            batched_data = batched_data.to(device)
            output = model(data, batched_data)
            data_list.extend(data.to("cpu"))
            mu = output["z_mu"]
            mu_list.extend(mu.detach().cpu().numpy())
            
    res = compute_batch_mixing_entropy(mu_list, np.array(batched_data_list))

    np.savez(f"./data/embeddings_90_{epoch}_hvae_gm_2_embedding.npy", embedding = mu_list, batches = np.array(batched_data_list), labels = np.array(label_data_list), or_data = np.array(data_list))
    
    print("knn result")
    print(res)
    return res


        
def main():


    cuda_id = 0
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    batch_size = 128*2
    learning_rate = 0.0001
    num_epochs = 50
    kld_weight = 0.00025
    latent_dim = 90

    train_data_name = "adata_ref_all_labels.npz"


    root_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = time.localtime()
    
    current_date_str = time.strftime(
        "%Y-%m-%d", current_time
    )
    
    current_time_str = time.strftime(
        "%H-%M-%S", current_time
    )
    
    experiment_dir = (
        f"{root_dir}/exp/vae/"
    )
    
    os.makedirs(experiment_dir, exist_ok=True)

    config = {
        "learning_rate": learning_rate,
        "architecture": "vae",
        "latent_dim": latent_dim,
        "dataset": train_data_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "kl_div_weight": kld_weight,
        "criteria": "",
        "optimizer": "Adam",
        "experiment_dir": experiment_dir,
        "seed": seed,
    }


    with open(f"{experiment_dir}/config.json", "w") as f:
        json.dump(config, f)

    train_dataset = SingleCellDataset2(
    data_path=f"{root_dir}/data/{train_data_name}"
    )

    arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
    encode_covariates=True,
    dropout_rate=0.2,
    n_layers=2,
    )
    
    model = HVAE_Lightning(n_input=10000, n_latent=latent_dim,n_hidden = 128, n_layers=1, encode_covariates=True, ln = False)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    dataloader2 = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )
   
    min_loss = np.inf
    best_epoch = 0
    val = 0
    rec_sum_res = 0
    
    from pytorch_lightning.callbacks import EarlyStopping

    early_stop_callback = EarlyStopping(
    monitor="train_loss",  
    patience=15,           
    mode="min",           
    verbose=True
    )


    trainer = Trainer(
    max_epochs=50, 
    callbacks=[early_stop_callback]
    )
    trainer.fit(model, dataloader, dataloader2)
    model.to("cpu")
    save_validate(dataloader2, 100, model)
    
        
if __name__ == "__main__":
    main()
