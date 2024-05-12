import numpy as np
import torch
from models.classifier import Classifier
from dataset.singlecell import SingleCellDataset
import wandb
import logging
from time import time
import os

logger = logging.getLogger("Simple-Classifier-Train")


def main():

    # Set experiment configurations
    cuda_id = 0
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    hidden_size = 1024
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 100
    eval_every = 10

    # Experiment directory based on timestamp
    root_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = f"{root_dir}/exp/classifier/{str(int(time()))}"
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="pineapple",
        # entity is specified by the username
        entity="acl21",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "simple-classifier",
            "dataset": "train90",
            "epochs": num_epochs,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "criteria": "CrossEntropyLoss",
            "optimizer": "Adam",
            "experiment_dir": experiment_dir,
        },
    )

    # Load the dataset
    train_dataset = SingleCellDataset(data_path=f"{root_dir}/data/train.npz")
    val_dataset = SingleCellDataset(data_path=f"{root_dir}/data/val.npz")

    # Define the model
    input_size = train_dataset.data.shape[-1]
    num_classes = len(train_dataset.label_to_index)
    model = Classifier(input_size, hidden_size, num_classes).to(device).float()

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

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
            inputs, targets = batch_data
            inputs = inputs.to(device).float()
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )
                # log metrics to wandb
                wandb.log({"epoch": epoch, "step": batch_idx, "loss": loss.item()})

            # Save the model
            if (epoch + 1) % eval_every == 0:
                # Validation loop with validation loss and accuracy
                val_loss = 0
                correct = 0
                total = 0
                model.eval()
                with torch.no_grad():
                    for val_inputs, val_targets in val_dataloader:
                        val_inputs = val_inputs.to(device)
                        val_targets = val_targets.to(device)
                        val_outputs = model(val_inputs)
                        val_loss += criterion(val_outputs, val_targets)
                        _, predicted = torch.max(val_outputs.data, 1)
                        total += val_targets.size(0)
                        correct += (predicted == val_targets).sum().item()

                    val_loss /= len(val_dataset)
                    print(
                        f"Val. Loss: {val_loss.item():.4f}, Val. Accuracy: {100*correct/total:.2f}%"
                    )
                    wandb.log(
                        {
                            "val_loss": val_loss.item(),
                            "val_accuracy": 100 * correct / total,
                        }
                    )
                model.train()

    # Save the final model
    torch.save(model.state_dict(), f"{experiment_dir}/model.pth")


if __name__ == "__main__":
    main()
