import torch
from torch import nn
from util.pickleInterface import load_codes, load_ret
from util.mnist.tools import read_mnist
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torchvision as torchv
import sys
from util.pytorchtools import EarlyStopping
from util.whatwhere.encoder import get_codes_examples


class MLP2H(nn.Module):
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


if __name__ == "__main__":
    """ ------------------------------------- """
    if len(sys.argv) != 2:
        print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
        exit(1)
    USE_WANDB = bool(int(sys.argv[1]))
    """ ------------------------------------- """
    # CODES
    param_id = "k20_Fs2_ep5_b0.8_Q21_Tw0.95"

    # MLP
    dim_hid = [2000]
    dim_in = 20 * 21 * 21
    dim_out = 28 * 28
    max_epochs = 100
    lr = 0.01
    batch_size = 100
    patience = 10
    delta = 0.001
    shuffle = False

    # MNIST
    trn_n = 600
    val_n = 100  # if (trn_n=60k;val_n=10k), then we will train with 50k and use 10k for valid

    imgs, _, _, _ = read_mnist()
    codes, _ = load_codes(param_id)

    imgs = imgs[:trn_n].reshape((trn_n, 28 * 28))
    codes = codes[:trn_n].toarray()

    dataset = TensorDataset(torch.Tensor(codes), torch.Tensor(imgs))

    trn_dataset, val_dataset = random_split(dataset, [trn_n - val_n, val_n])

    trn_loader = DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP2H(input_dim=dim_in, output_dim=dim_out, hidden_dim=dim_hid).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    if USE_WANDB:
        wandb.init(
            project="MLP_small",
            entity="rodrigosimass",
            config={
                "n_hLayers": len(dim_hid),
                "n_hUnits": dim_hid,
                "lr": lr,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "shuffle_data": shuffle,
                "patience": patience,
                "trn_n": trn_n,
            },
        )

    if patience > 0:
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta)
    for epoch in range(0, max_epochs):
        print(f"Starting epoch {epoch+1}")

        # Training
        trn_loss = 0.0
        for inputs, targets in val_loader:

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Perform forward pass

            # Compute loss
            loss = loss_function(outputs, targets)
            trn_loss += loss.item()

            loss.backward()  # Perform backward pass
            optimizer.step()  # Perform optimization
        trn_loss = trn_loss / len(trn_loader)

        # Validation
        val_loss = 0.0
        for inputs, targets in val_loader:
            outputs = model(inputs)  # Forward Pass
            loss = loss_function(outputs, targets)
            val_loss += loss.item()
        val_loss = val_loss / len(val_loader)

        print(f"MSE Loss (train) = {trn_loss} ; MSE Loss (validation) = {val_loss}")

        # Early Stopping
        if patience > 0:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Log stats
        if USE_WANDB:

            # ex_target = val_dataset.dataset.tensors[1][-10:].reshape((10, 28, 28))
            ex_target = targets[-10:].reshape((10, 28, 28))
            tensor = torch.unsqueeze(ex_target, dim=1)
            grid = torchv.utils.make_grid(tensor, normalize=True, nrow=10, pad_value=1)
            ex_target = wandb.Image(grid)

            """ optimizer.zero_grad()  # Zero the gradients
            ex_out = model(val_dataset.dataset.tensors[0][-10:]).reshape(
                (10, 28, 28)
            )  # forward the examples """
            ex_out = outputs[-10:].reshape((10, 28, 28))
            tensor = torch.unsqueeze(ex_out, dim=1)
            grid = torchv.utils.make_grid(tensor, normalize=True, nrow=10, pad_value=1)
            ex_out = wandb.Image(grid)

            wandb.log(
                {
                    "MSE_train": trn_loss / len(trn_loader),
                    "MSE_valid": val_loss / len(val_loader),
                    "n_epochs": epoch,
                    "Output": ex_out,
                    "Target": ex_target,
                },
                step=epoch,
            )

    print("Training done...")
    if USE_WANDB:
        wandb.finish()
