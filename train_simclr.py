import numpy as np
import torch
from models.encoder import Encoder
from dataset.singlecell import SingleCellDataset
import wandb
import logging
import time
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
from torch.utils.data import TensorDataset


def main(train_data_name, embedding_save, latent_dim = 1024, learning_rate = 0.0001, batch_size = 32, num_epochs = 100, eval_every = 1, print_every = 500, temperature = 0.5):

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = SingleCellDataset(
        data_path=train_data_name",
        sample_fraction=0.1
    )

    #### use full for plotting
    val_dataset = SingleCellDataset(
        data_path=train_data_name,
        sample_fraction=1.0
    )
    
    input_size = train_dataset.data.shape[-1]
    model = Encoder(input_size, latent_dim=latent_dim).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
         val_dataset, batch_size=batch_size, shuffle=False
    )


    #-----------------------------------------------------------------------------

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(dataloader):
            data, labels = batch_data
            data = data.to(device).float()

            optimizer.zero_grad()


            z = model(data)
            loss = nt_xent_loss(z, labels, temperature=temperature)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

        if (epoch + 1) % eval_every == 0:
            torch.save(model.state_dict(), f"{experiment_dir}/model_{epoch}.pth")


    model.eval()
    embeddings = []
    labels_train = []

    with torch.no_grad():
        for data, labels in val_dataloader:
            data = data.to(device).float()
            z = model(data)
            embeddings.append(z.cpu().numpy())
            labels_train.extend(labels.numpy())

    np.savez(embedding_save, embeddings = embeddings, labels = labels_train)

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
    main()
