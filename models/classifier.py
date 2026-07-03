import torch
import torch.nn as nn


class Classifier(nn.Module):
    """A single-hidden-layer MLP classifier over precomputed embeddings."""

    def __init__(self, input_size, hidden_size, num_classes):
        """
        Args:
            input_size (int): Dimensionality of the input embedding.
            hidden_size (int): Width of the hidden layer.
            num_classes (int): Number of output classes.
        """
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, only_representations=False):
        """
        Args:
            x (Tensor): Input embeddings, shape (batch, input_size).
            only_representations (bool, optional): If True, return the hidden
                representation instead of class logits. Defaults to False.

        Returns:
            Tensor: Class logits, or the hidden representation if
                `only_representations=True`.
        """
        out = self.fc1(x)
        out = self.relu1(out)
        if only_representations:
            return out
        out = self.fc3(out)
        return out

    def get_representation(self, x):
        """Returns the hidden-layer representation of `x` (equivalent to `forward(x, only_representations=True)`)."""
        out = self.fc1(x)
        out = self.relu1(out)
        return out
