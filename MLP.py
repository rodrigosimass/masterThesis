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


class MLP2H(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.decoder_h1 = nn.Linear(in_features=20 * 21 * 21, out_features=4000)
        self.decoder_h2 = nn.Linear(in_features=4000, out_features=2000)
        self.decoder_out = nn.Linear(in_features=2000, out_features=28 * 28)

    def forward(self, features):
        activation = self.decoder_h1(features)
        activation = torch.relu(activation)
        code = self.decoder_h2(activation)
        code = torch.relu(code)
        activation = self.decoder_out(code)
        reconstructed = torch.relu(activation)
        return reconstructed


if __name__ == "__main__":
    """ ------------------------------------- """
    if len(sys.argv) != 2:
        print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
        exit(1)
    USE_WANDB = bool(sys.argv[1])
    """ ------------------------------------- """
    param_id = "k20_Fs2_ep5_b0.8_Q21_Tw0.95"
    n_epochs = 100
    lr = 0.0001
    batch_size = 100
    shuffle = False

    if USE_WANDB:
        wandb.init(
            project="MLP_small",
            entity="rodrigosimass",
            config={
                "n_hLayers": MLP2H.modules,
                "n_hUnits": 6000,
                "lr": lr,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "shuffle_data": shuffle,
            },
        )

    imgs, _, _, _ = read_mnist()
    codes, _ = load_codes(param_id)

    imgs = imgs[:600].reshape((600, 28 * 28))
    codes = codes[:600].toarray()

    trn_codes, val_codes = random_split(codes, [500, 100])
    trn_imgs, val_imgs = random_split(imgs, [500, 100])

    tensor_ti = torch.Tensor(trn_imgs)
    tensor_tc = torch.Tensor(trn_codes)
    tensor_vi = torch.Tensor(val_imgs)
    tensor_vc = torch.Tensor(val_codes)

    trn_dataset = TensorDataset(tensor_tc, tensor_ti)
    val_dataset = TensorDataset(tensor_vc, tensor_vi)

    trn_loader = DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP2H().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    for epoch in range(0, n_epochs):
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

        # Validation
        val_loss = 0.0
        for inputs, targets in val_loader:
            outputs = model(inputs)  # Forward Pass
            loss = loss_function(outputs, targets)
            val_loss += loss.item()

        print(
            f"MSE Loss (train) = {trn_loss}/{len(trn_loader)} = {trn_loss / len(trn_loader)}"
        )
        print(
            f"MSE Loss (validation) = {val_loss}/{len(val_loader)} = {val_loss / len(val_loader)}"
        )

        if USE_WANDB:

            orig = val_imgs[-10:].reshape((10, 28, 28))
            tensor = torch.from_numpy(orig)
            tensor = torch.unsqueeze(tensor, dim=1)
            grid = torchv.utils.make_grid(tensor, normalize=True, nrow=10, pad_value=1)
            orig = wandb.Image(grid)

            recon = np.array(outputs.detach())[-10:].reshape((10, 28, 28))
            tensor = torch.from_numpy(recon)
            tensor = torch.unsqueeze(tensor, dim=1)
            grid = torchv.utils.make_grid(tensor, normalize=True, nrow=10, pad_value=1)
            recon = wandb.Image(grid, caption=f"Epoch {epoch+1}")

            wandb.log(
                {
                    "MSE_train": trn_loss / len(trn_loader),
                    "MSE_valid": val_loss / len(val_loader),
                    "originals": orig,
                    "reconstructions": recon,
                },
                step=epoch,
            )

            wandb.log({"originals": orig})

    print("Training done...")
    if USE_WANDB:
        wandb.finish()
