import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        y = torch.relu(self.layers[-1](x))
        return y


class pseudoWN(nn.Module):
    def __init__(self, pattern_size, W, b):
        super().__init__()
        self.fc = nn.Linear(pattern_size, pattern_size)
        self.tanh = torch.nn.Tanh()
        self.fc.weight = W
        self.fc.bias = b

    def forward(self, x):
        y = self.fc(x.float())
        out = self.tanh(y)
        sign = torch.sign(out)
        binary_out = torch.relu(sign)
        return binary_out