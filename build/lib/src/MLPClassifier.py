import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()

    def reset_weights(self):
        for layer in self.model:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()