import torch
from torch import nn


def loss_function(name):
    if name == "MSE":
        return nn.MSELoss()
    elif name == "L1":
        return nn.L1Loss()
    elif name == "KLDiv":
        return nn.KLDivLoss()
    else:
        print(f"Error! unknown loss function: {name}")
        exit(0)


if __name__ == "__main__":
    m = nn.LogSoftmax()
    
    input = torch.randn(2, 3)
    output = m(input)

    print(input)
    print(output)