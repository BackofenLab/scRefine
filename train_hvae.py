import numpy as np
import torch
from models.vae import HVAE_Lightning7
from dataset.singlecell import SingleCellDataset
import logging
import time
import os
import json
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from pytorch_lightning import Trainer
import random
from pytorch_lightning.callbacks import EarlyStopping
    
def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    

def save_validate(dataloader, model, save_embedding):
    cuda_id = 0
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
            label_data_list.extend(np.array(label_data))
            batched_data = batched_data.to(device)
            output = model(data, gene, batched_data)
            data_list.extend(data.to("cpu"))
            mu = output["z_mu"]
            mu_list.extend(mu.detach().cpu().numpy())
        
    np.savez(save_embedding, embedding = mu_list, batches = np.array(batched_data_list), labels = np.array(label_data_list), or_data = np.array(data_list))

    return 


        
def main(train_data_name, model_checkpoint, val_data_path = None, num_epochs = 50, latent_dim = 90, hidden_dim = 128, layers = 1, learning_rate = 0.0001, batch_size = 256, patience = 15, encode_covariates =True, ln = False, gene_factor = 20, save_embedding = "embedding.npz", model_checkpoint = "model.ckpt"):


    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = SingleCellDataset(
    data_path=train_data_name
    )
    
    if val_data_path != None:
        val_dataset = SingleCellDataset(
        data_path=val_data_path
        )
    

    model = HVAE_Lightning7(n_input=train_dataset.return_size(), n_latent=latent_dim,n_hidden = hidden_dim, n_layers=layers, encode_covariates=encode_covariates, ln = ln, num_components = train_dataset.return_num_batches(), gene_factor = gene_factor, prediction_genes = train_dataset.num_marker_genes())
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    if val_data_path != None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
    else:
        val_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )

    early_stop_callback = EarlyStopping(
    monitor="train_loss",  
    patience=patience,          
    mode="min",            
    verbose=True
    )

    trainer = Trainer(
    max_epochs=num_epochs, 
    callbacks=[early_stop_callback]
    )
    trainer.fit(model, dataloader, val_dataloader)
    trainer.save_checkpoint(model_checkpoint)
    
    model.to("cpu")
    save_validate(val_dataloader, model, save_embedding)
    

        
if __name__ == "__main__":
    set_seed()
    main()
    
