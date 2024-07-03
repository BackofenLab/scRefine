import numpy as np
import torch
from models.vae_toy import VAE
from dataset.toy import ToyDataset
import wandb
import logging
import time
import os
import json

logger = logging.getLogger("VAE-Toy-Train")


def main():

    # Set experiment configurations
    cuda_id = 2
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    latent_dim = 32
    learning_rate = 0.0001
    batch_size = 64
    num_epochs = 30
    eval_every = 10
    print_every = 100
    kld_weight = 0.00025
    train_data_name = "toy/train_toy_2blobs"

    # Experiment directory based on timestamp
    root_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = time.localtime()
    current_date_str = time.strftime(
        "%Y-%m-%d", current_time
    )  # Current date in YYYY-MM-DD format
    current_time_str = time.strftime(
        "%H-%M-%S", current_time
    )  # Current time in HH:MM:SS format
    experiment_dir = f"{root_dir}/exp/vae-toy/{current_date_str}/{current_time_str}/"
    os.makedirs(experiment_dir, exist_ok=True)

    # Hyperparameters
    config = {
        "learning_rate": learning_rate,
        "architecture": "vae-toy",
        "dataset": train_data_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "criteria": "MSELoss",
        "optimizer": "Adam",
        "kl_div_weight": kld_weight,
        "experiment_dir": experiment_dir,
        "seed": seed,
    }
    # Initialize wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="pineapple",
        # entity is specified by the username
        entity="acl21",
        # track hyperparameters and run metadata
        config=config,
    )
    # Save hyperparameters to json file
    with open(f"{experiment_dir}/config.json", "w") as f:
        json.dump(config, f)

    # Load the dataset
    train_dataset = ToyDataset(data_path=f"{root_dir}/data/{train_data_name}.npz")

    # Define the model
    input_size = train_dataset.data.shape[-1]
    model = VAE(input_size, latent_dim=latent_dim).to(device).float()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define the data loader
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(dataloader):
            data, _ = batch_data
            data = data.to(device).float()

            optimizer.zero_grad()

            # Forward pass
            recon_data, mu, log_var = model(data)
            recon_loss = torch.nn.functional.mse_loss(recon_data, data)
            kl_div = torch.mean(
                -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
            )

            # Backward pass
            loss = recon_loss + kld_weight * kl_div
            loss.backward()
            optimizer.step()

            # Logging
            if (batch_idx + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Recon Loss: {recon_loss.detach().item():.4f}, KL Div: {-kl_div.detach().item():.4f}"
                )
                # log metrics to wandb
                wandb.log(
                    {
                        "epoch": epoch,
                        "step": batch_idx,
                        "loss": loss.item(),
                        "recon_loss": recon_loss.detach().item(),
                        "kl_div": -kl_div.detach().item(),
                    }
                )
        if (epoch + 1) % eval_every == 0:
            # Save the final model
            torch.save(model.state_dict(), f"{experiment_dir}/model_{epoch}.pth")


if __name__ == "__main__":
    main()
