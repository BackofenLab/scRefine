import numpy as np
import torch
from models.classifier3 import Classifier
from dataset.singlecell import SingleCellDataset
from sklearn.metrics import classification_report
import wandb
import logging
import time
import os
import json

logger = logging.getLogger("Simple-Classifier-Train")


def main():

    # Set experiment configurations
    cuda_id = 0
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    hidden_size = 1024
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
    experiment_dir = f"{root_dir}/exp/classifier/{current_date_str}/{current_time_str}/"
    os.makedirs(experiment_dir, exist_ok=True)

    # Hyperparameters
    config = {
        "learning_rate": learning_rate,
        "architecture": "simple-classifier-3-layers",
        "dataset": train_data_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "criteria": "CrossEntropyLoss",
        "optimizer": "Adam",
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
            if (batch_idx + 1) % 1000 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )
                # log metrics to wandb
                wandb.log({"epoch": epoch, "step": batch_idx, "loss": loss.item()})

        # Validation loop
        if (epoch + 1) % eval_every == 0:
            val_loss = 0
            correct = 0
            total = 0
            model.eval()
            pred_labels = []
            true_labels = []
            with torch.no_grad():
                for val_inputs, val_targets in val_dataloader:
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs, val_targets)
                    _, predicted = torch.max(val_outputs.data, 1)
                    total += val_targets.size(0)
                    correct += (predicted == val_targets).sum().item()
                    pred_labels.extend(predicted.cpu().numpy().tolist())
                    true_labels.extend(val_targets.cpu().numpy().tolist())

                val_loss /= len(val_dataset)
                print(
                    f"Val. Loss: {val_loss.item():.4f}, Val. Accuracy: {100*correct/total:.2f}%"
                )
                wandb.log(
                    {
                        "val_loss": val_loss.item(),
                        "val_accuracy": 100 * correct / total,
                        "epoch": epoch,
                    }
                )
                # Save the classification report in a text file
                report = classification_report(
                    true_labels,
                    pred_labels,
                    target_names=list(val_dataset.label_to_index.keys()),
                )
                with open(f"{experiment_dir}/cls_report_{epoch+1}.txt", "w") as f:
                    f.write(report)

            model.train()

    # Save the final model
    torch.save(model.state_dict(), f"{experiment_dir}/model.pth")


if __name__ == "__main__":
    main()
