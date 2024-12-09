import numpy as np
import torch
from models.vae_small import VAE
from dataset.singlecell_marker import SingleCellDataset
import wandb
import logging
import time
import os
import json
import argparse

logger = logging.getLogger("VAE-Train")


def main(args):

    # Set experiment configurations
    cuda_id = args.cuda
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    latent_dim = args.latent_dim
    learning_rate = 0.0001
    batch_size = 32
    num_epochs = 50
    eval_every = 10
    print_every = 1000
    kld_weight = 0.00025
    # train_data_name = "gpt_classes/data_top_5000_genes"
    # train_data_name = "sven_classes/data_high_top_10000_genes_sparse_markers"
    # val_data_name = "sven_classes/val_high_10"
    # train_data_name = "Predicted_labels_CellTypist/t_cells/data_t_cells_top_12000_genes"
    # train_data_name = (
    # "Predicted_labels_CellTypist/data_low_top_10000_genes_sparse_markers"
    # )
    # val_data_name = "Predicted_labels_CellTypist/val_low_10"
    train_data_name = "gpt_classes/data_top_10000_genes_sparse_markers"
    markers_name = args.markers

    # Experiment directory based on timestamp
    root_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = time.localtime()
    current_date_str = time.strftime(
        "%Y-%m-%d", current_time
    )  # Current date in YYYY-MM-DD format
    current_time_str = time.strftime(
        "%H-%M-%S", current_time
    )  # Current time in HH:MM:SS format
    experiment_dir = (
        f"{root_dir}/exp/vae/{latent_dim}/{current_date_str}/{current_time_str}/"
    )
    os.makedirs(experiment_dir, exist_ok=True)

    # Hyperparameters
    config = {
        "learning_rate": learning_rate,
        "architecture": "vae",
        "latent_dim": latent_dim,
        "dataset": train_data_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "kl_div_weight": kld_weight,
        "criteria": "MSELoss",
        "optimizer": "Adam",
        "experiment_dir": experiment_dir,
        "seed": seed,
        "markers": markers_name,
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
    train_dataset = SingleCellDataset(
        data_path=f"{root_dir}/data/{train_data_name}.npz", markers_name=markers_name
    )
    # val_dataset = SingleCellDataset(data_path=f"{root_dir}/data/{val_data_name}.npz")

    # Save label to index mapping
    with open(f"{experiment_dir}/label_to_index.json", "w") as f:
        json.dump(train_dataset.label_to_index, f)

    # Define the model
    input_size = sum(train_dataset.marker_genes_indices)
    model = VAE(input_size, latent_dim=latent_dim).to(device).float()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define the data loader
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=True
    # )

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(dataloader):
            data, _ = batch_data
            data = data.to(device).float()

            optimizer.zero_grad()

            # Forward pass
            recon_data, mu, logvar = model(data)

            # Backward pass
            loss_dict = model.loss_function(recon_data, data, mu, logvar)
            loss_dict["loss"].backward()
            optimizer.step()

            # Logging
            if (batch_idx + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss_dict['loss'].item():.4f}, Recon Loss: {loss_dict['recon_loss'].item():.4f}, KLD Loss: {loss_dict['kld_loss'].item():.4f}"
                )
                # log metrics to wandb
                wandb.log(
                    {
                        "epoch": epoch,
                        "step": batch_idx,
                        "loss": loss_dict["loss"].item(),
                        "recon_loss": loss_dict["recon_loss"].item(),
                        "kld_loss": loss_dict["kld_loss"].item(),
                    }
                )
        if (epoch + 1) % eval_every == 0:
            # Save the final model
            torch.save(model.state_dict(), f"{experiment_dir}/model_{epoch}.pth")


if __name__ == "__main__":

    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent_dim", type=int, default=2, help="Dimension of the latent space"
    )
    parser.add_argument("--cuda", type=int, default=1, help="CUDA Device ID")
    parser.add_argument(
        "--markers", type=str, default="markers_10", help="Name of the marker genes"
    )
    args = parser.parse_args()
    main(args)
