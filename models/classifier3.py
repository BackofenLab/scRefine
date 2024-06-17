import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, only_representations=False):
        out = self.fc1(x)
        out = self.relu1(out)
        if only_representations:
            return out
        out = self.fc3(out)
        return out

    def get_representation(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        return out
