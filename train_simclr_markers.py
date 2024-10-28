import numpy as np
import torch
from models.encoder import Encoder
from dataset.singlecell_marker import SingleCellDataset
import wandb
import logging
import time
import os
import json

logger = logging.getLogger("SimCLR-Train")


def main():

    # Set experiment configurations
    cuda_id = 1
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    latent_dim = 1024
    learning_rate = 0.0001
    batch_size = 32
    num_epochs = 50
    eval_every = 10
    print_every = 500
    temperature = 0.5
    # train_data_name = "sven_classes/data_high_top_10000_genes"
    # val_data_name = "sven_classes/val_high_10"
    # train_data_name = "Predicted_labels_CellTypist/t_cells/data_t_cells_top_10000_genes"
    train_data_name = (
        "Predicted_labels_CellTypist/data_low_top_10000_genes_sparse_markers"
    )
    # val_data_name = "Predicted_labels_CellTypist/val_low_10"

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
        f"{root_dir}/exp/simclr/{latent_dim}/{current_date_str}/{current_time_str}/"
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
        "temperature": temperature,
        "criteria": "NT-XentLoss",
        "optimizer": "Adam",
        "experiment_dir": experiment_dir,
        "seed": seed,
        "comment": "Using marker genes for positive pairs",
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
        data_path=f"{root_dir}/data/{train_data_name}.npz"
    )
    # val_dataset = SingleCellDataset(data_path=f"{root_dir}/data/{val_data_name}.npz")

    # Save label to index mapping
    with open(f"{experiment_dir}/label_to_index.json", "w") as f:
        json.dump(train_dataset.label_to_index, f)

    # Define the model
    input_size = train_dataset.data.shape[-1]
    model = Encoder(input_size, latent_dim=latent_dim).to(device).float()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define the data loader
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=True
    # )

    marker_indices = torch.from_numpy(train_dataset.marker_genes_indices).to(device)

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(dataloader):
            data, _ = batch_data
            data = data.to(device).float()

            optimizer.zero_grad()

            # Forward pass
            z = model(data)

            loss = nt_xent_loss(z, data, marker_indices, temperature)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Logging
            if (batch_idx + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )
                # log metrics to wandb
                wandb.log(
                    {
                        "epoch": epoch,
                        "step": batch_idx,
                        "loss": loss.item(),
                    }
                )
        if (epoch + 1) % eval_every == 0:
            # Save the final model
            torch.save(model.state_dict(), f"{experiment_dir}/model_{epoch}.pth")


# SimCLR-like loss: NT-Xent Loss (with ground truth labels for positive pairs)
def nt_xent_loss(z, data, marker_indices, temperature=0.5):
    batch_size = z.shape[0]

    # Normalize the representations
    z = torch.nn.functional.normalize(z, dim=1)

    # Cosine similarity matrix between all elements in the batch
    similarity_matrix = torch.matmul(z, z.T)

    # Scale the similarity matrix with the temperature
    similarity_matrix = similarity_matrix / temperature

    # Create a soft mask for positive pairs based on the cosine similarity of marker genes
    data_filtered = torch.nn.functional.normalize(data[:, marker_indices], dim=1)
    positive_mask = data_filtered @ data_filtered.T

    # Set positive pairs to 1 and the rest to 0
    positive_mask = positive_mask > 0.75

    # Mask out the diagonal (self-contrast, i.e., a sample shouldn't compare to itself)
    mask = torch.eye(batch_size, dtype=torch.bool).to(similarity_matrix.device)

    # For stability: subtract the maximum value for each row before exponentiating
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Calculate softmax scores (exp/log sums) for the denominator, excluding self-similarities
    exp_logits = torch.exp(logits) * (~mask)  # Zero out diagonal instead of -inf
    exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

    # Calculate log-probabilities for positive pairs
    log_prob = logits - torch.log(exp_logits_sum + 1e-10)

    # Select the log-probabilities of the positive pairs (where positive_mask == 1)
    positive_log_prob = (positive_mask * log_prob).sum(dim=1) / (
        positive_mask.sum(dim=1) + 1e-10
    )

    # Loss is the mean negative log likelihood of the positive pairs
    loss = -positive_log_prob.mean()

    return loss


if __name__ == "__main__":
    main()
