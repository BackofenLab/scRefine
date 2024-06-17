import numpy as np
import torch
from models.vae import SCPHERE
from dataset.singlecell import SingleCellDataset
import wandb
import logging
import time
import os
import json

logger = logging.getLogger("SCPHERE-VAE-Train")


def main():

    # Set experiment configurations
    cuda_id = 0
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    latent_dim = 32
    learning_rate = 0.0001
    batch_size = 32
    num_epochs = 10
    eval_every = 10
    train_data_name = "train_high_50"
    val_data_name = "val_high_50"

    # Experiment directory based on timestamp
    root_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = time.localtime()
    current_date_str = time.strftime(
        "%Y-%m-%d", current_time
    )  # Current date in YYYY-MM-DD format
    current_time_str = time.strftime(
        "%H-%M-%S", current_time
    )  # Current time in HH:MM:SS format
    experiment_dir = f"{root_dir}/exp/vae/{current_date_str}/{current_time_str}/"
    os.makedirs(experiment_dir, exist_ok=True)

    # Hyperparameters
    config = {
        "learning_rate": learning_rate,
        "architecture": "scphere-vae",
        "dataset": train_data_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "criteria": "BCEWithLogitsLoss",
        "optimizer": "Adam",
        "experiment_dir": experiment_dir,
        "seed": seed,
    }
    # Initialize wandb
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="pineapple",
    #     # entity is specified by the username
    #     entity="acl21",
    #     # track hyperparameters and run metadata
    #     config=config,
    # )
    # Save hyperparameters to json file
    with open(f"{experiment_dir}/config.json", "w") as f:
        json.dump(config, f)

    # Load the dataset
    train_dataset = SingleCellDataset(
        data_path=f"{root_dir}/data/{train_data_name}.npz"
    )
    val_dataset = SingleCellDataset(data_path=f"{root_dir}/data/{val_data_name}.npz")

    # Save label to index mapping
    with open(f"{experiment_dir}/label_to_index.json", "w") as f:
        json.dump(train_dataset.label_to_index, f)

    # Define the model
    input_size = train_dataset.data.shape[-1]
    num_classes = len(train_dataset.label_to_index)
    model = (
        SCPHERE(input_size, latent_dim=latent_dim, distribution="vmf")
        .to(device)
        .float()
    )

    # Define the loss function
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define the data loader
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(dataloader):
            data, _ = batch_data
            data = data.to(device).float()

            optimizer.zero_grad()

            # Forward pass
            _, (q_z, p_z), _, data_recon = model(data)
            recon_loss = criterion(data_recon, data).sum(-1).mean()
            if model.distribution == "normal":
                kl_div = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
            elif model.distribution == "vmf":
                kl_div = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            else:
                raise NotImplemented

            # Backward pass
            loss = recon_loss + kl_div
            loss.backward()
            optimizer.step()

            # Logging
            if (batch_idx + 1) % 1000 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_div.item():.4f}"
                )
                # log metrics to wandb
                # wandb.log(
                #     {
                #         "epoch": epoch,
                #         "step": batch_idx,
                #         "loss": loss.item(),
                #         "recon_loss": recon_loss.item(),
                #         "kl_div": kl_div.item(),
                #     }
                # )

    # Save the final model
    torch.save(model.state_dict(), f"{experiment_dir}/model.pth")


if __name__ == "__main__":
    main()
