import torch
from torch import nn
from util.pickleInterface import get_codes_run_name, load_codes
from util.mnist.tools import idxs_x_random_per_class, read_mnist
from torch.utils.data import TensorDataset, DataLoader, random_split
from util.pytorch.print_log import print_model_info
import wandb
import torchvision as torchv
import sys
from util.pytorch.earlystopping import *
from models.MLP import *

if __name__ == "__main__":
    """-------------------------------------"""
    if len(sys.argv) != 2:
        print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
        exit(1)
    USE_WANDB = bool(int(sys.argv[1]))
    """ ------------------------------------- """

    # CODES
    k = 20
    n_epochs = 5
    b = 0.8
    Q = 21
    Fs = 2
    Tw = 0.95

    code_id = get_codes_run_name(k, Fs, n_epochs, b, Q, Tw)

    # MLP
    dim_hid = [200, 200]
    dim_in = 20 * 21 * 21
    dim_out = 28 * 28

    shuffle = True
    lr = 1e-3

    max_epochs = 300
    patience = 5
    delta = 1e-4

    # Dataset sizes
    batch_size = 4
    size = 5000

    trn_n = 6 * size
    val_n = 2 * size
    tst_n = 1 * size

    tst_model = True
    save_model = True

    model_name = "MLP_decoder" + str(dim_hid) + "_n_" + str(trn_n)

    ret_percent = 0

    trn_imgs, _, tst_imgs, tst_lbls = read_mnist()
    trn_imgs = trn_imgs[:trn_n].reshape((trn_n, 28 * 28))
    tst_imgs = tst_imgs[:tst_n].reshape((tst_n, 28 * 28))
    tst_lbls = tst_lbls[:tst_n]

    trn_codes, tst_codes = load_codes(code_id)
    trn_codes = trn_codes[:trn_n].toarray()
    tst_codes = tst_codes[:tst_n].toarray()

    # create a tensor from the test set for each class (to visualize reconstructions)
    idxs = idxs_x_random_per_class(tst_lbls)
    tst_in = torch.from_numpy(tst_codes[idxs])
    tst_target = torch.from_numpy(tst_imgs[idxs])
    tst_target = tst_target.reshape((-1, 28, 28))
    tst_target = torch.unsqueeze(tst_target, dim=1)  # Add the channel dimention

    trn_dataset = TensorDataset(torch.Tensor(trn_codes), torch.Tensor(trn_imgs))
    tst_dataset = TensorDataset(torch.Tensor(tst_codes), torch.Tensor(tst_imgs))

    trn_dataset, val_dataset = random_split(trn_dataset, [trn_n - val_n, val_n])

    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    tst_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=shuffle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    model = MLP(input_dim=dim_in, output_dim=dim_out, hidden_dim_list=dim_hid).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    print_model_info(model, optimizer, loss_function, verbose=2)

    if USE_WANDB:
        wandb.init(
            project="decoder_mlp",
            entity="rodrigosimass",
            config={
                "ww_k": k,
                "ww_Fs": Fs,
                "ww_n_epochs": n_epochs,
                "ww_b": b,
                "ww_Q": Q,
                "ww_Tw": Tw,
                "MLP_n_layers": len(dim_hid),
                "MLP_n_units": dim_hid,
                "ann_lr": lr,
                "ann_max_epochs": max_epochs,
                "ann_batch_size": batch_size,
                "ann_shuffle_data": shuffle,
                "ann_patience": patience,
                "ann_delta": delta,
                "ann_loss": str(loss_function),
                "trn_n": trn_n,
            },
        )

    if patience > 0:
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            delta=delta,
            name=model_name,
            save_model=save_model,
        )
    for epoch in range(0, max_epochs):
        print(f"Starting epoch {epoch}")

        # Training
        trn_loss = 0.0
        for inputs, targets in trn_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

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
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)  # Forward Pass
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)

        # Early Stopping
        if patience > 0:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Log stats
        if USE_WANDB:
            log_dict = {"trn_loss": trn_loss, "val_loss": val_loss}

            if epoch == 0:
                # Example grid or target (desired outputs)
                grid = torchv.utils.make_grid(
                    tst_target, normalize=True, nrow=10, pad_value=1, range=(0, 1)
                )
                ex_target = wandb.Image(grid)
                log_dict["Target"] = ex_target

            if (
                early_stopping.counter == 0 or epoch == 0
            ):  # Improved -> log reconstructions
                with torch.no_grad():
                    tst_in = tst_in.to(device)
                    tst_out = model(tst_in.float())
                tst_out = tst_out.reshape((-1, 28, 28))
                tst_out = torch.unsqueeze(tst_out, dim=1)  # Add empty channel dimention
                grid = torchv.utils.make_grid(
                    tst_out, normalize=True, nrow=10, pad_value=1, range=(0, 1)
                )
                ex_out = wandb.Image(grid)
                log_dict["Output"] = ex_out

            wandb.log(log_dict, step=epoch)

    print("Training done...")

    if tst_model:
        tst_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tst_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)  # Forward Pass
                loss = loss_function(outputs, targets)
                tst_loss += loss.item()
        tst_loss = tst_loss / len(tst_loader)

        if USE_WANDB:
            wandb.log({"tst_loss": tst_loss})

        print(f"Test set avg. MSE loss = {tst_loss}")

    if USE_WANDB:
        wandb.finish()
