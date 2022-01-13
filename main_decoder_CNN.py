import wandb
import sys
import torch
import torchvision as torchv
from torch.utils.data import TensorDataset, DataLoader, random_split
from util.pickleInterface import *
from util.mnist.tools import *
from util.pytorch.earlystopping import *
from util.pytorch.print_log import print_model_info
from util.pytorch.tools import *
from models.CNN import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
        exit(1)
    USE_WANDB = bool(int(sys.argv[1]))

    """ ---------- PARAMS ---------- """
    # CODES
    rng = np.random.RandomState(0)  # reproducible
    k = 30
    Fs = 2
    n_epochs = 5
    b = 0.8
    Q = 18
    Tw = 0.6
    wta = True

    # early stopping
    patience = 5
    delta = 1e-3

    # train
    max_epochs = 100
    lr = 1e-2

    # Dataset sizes
    batch_size = 8

    trn_n = 50000
    val_n = 10000
    tst_n = 8000

    tst_model = True
    save_model = False

    model_name = "CNN" + "_n_" + str(trn_n)

    trn_imgs, trn_lbls, tst_imgs, tst_lbls = read_mnist()

    features = compute_features(
        trn_imgs, trn_lbls, k, Fs, rng, n_epochs, b, verbose=True
    )

    trn_codes, _ = compute_codes(
        trn_imgs,
        k,
        Q,
        features,
        Tw,
        wta,
        n_epochs,
        b,
        Fs,
        verbose=True,
        set="trn",
    )
    tst_codes, _ = compute_codes(
        tst_imgs,
        k,
        Q,
        features,
        Tw,
        wta,
        n_epochs,
        b,
        Fs,
        verbose=True,
        set="tst",
    )

    trn_imgs = trn_imgs[:trn_n].reshape((trn_n, 28 * 28))
    tst_imgs = tst_imgs[:tst_n].reshape((tst_n, 28 * 28))
    tst_lbls = tst_lbls[:tst_n]

    features = torch.from_numpy(features)
    features = torch.unsqueeze(features, dim=1)

    trn_codes = swap_codes_axes(trn_codes[:trn_n], k, Q)  # (-1,Q,Q,K) -> (-1,K,Q,Q)
    tst_codes = swap_codes_axes(tst_codes[:tst_n], k, Q)  # (-1,Q,Q,K) -> (-1,K,Q,Q)

    # create a tensor from the test set for each class (to visualize reconstructions)
    idxs = idxs_x_random_per_class(tst_lbls)
    tst_in = torch.from_numpy(tst_codes[idxs])
    tst_target = torch.from_numpy(tst_imgs[idxs])
    tst_target = tst_target.reshape((-1, 28, 28))
    tst_target = torch.unsqueeze(tst_target, dim=1)  # Add the channel dimention

    trn_dataset = TensorDataset(torch.Tensor(trn_codes), torch.Tensor(trn_imgs))
    tst_dataset = TensorDataset(torch.Tensor(tst_codes), torch.Tensor(tst_imgs))

    trn_dataset, val_dataset = random_split(trn_dataset, [trn_n - val_n, val_n])

    trn_loader = DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    tst_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    model = CNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_function = nn.MSELoss()

    """if init_kernels:
        model.state_dict()["convT1.weight"][:] = features"""

    if USE_WANDB:
        wandb.init(
            project="decoder_cnn",
            entity="rodrigosimass",
            config={
                "ww_k": k,
                "ww_Fs": Fs,
                "ww_n_epochs": n_epochs,
                "ww_b": b,
                "ww_Q": Q,
                "ww_Tw": Tw,
                "ann_lr": lr,
                "ann_max_epochs": max_epochs,
                "ann_batch_size": batch_size,
                "ann_patience": patience,
                "ann_trn_n": trn_n,
                "ann_val_n": val_n,
                "ann_loss": str(loss_function),
            },
        )

    print_model_info(model, optimizer, loss_function, verbose=1)

    if patience > 0:
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            delta=delta,
            name=model_name,
            save_model=save_model,
        )
    for epoch in range(0, max_epochs):
        print(f"Starting epoch {epoch+1}")

        # Training
        trn_loss = 0.0
        for inputs, targets in trn_loader:

            model.train()

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Perform forward pass

            # Compute loss
            loss = loss_function(outputs, targets)
            trn_loss += loss.item()

            loss.backward()  # Perform backward pass
            optimizer.step()  # Perform optimization
        avg_trn_loss = trn_loss / len(trn_loader)

        # Validation
        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)  # Forward Pass
                loss = loss_function(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # print(f"Loss (train) = {avg_trn_loss} ; Loss (validation) = {val_loss}")

        # Early Stopping
        if patience > 0:
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Log stats
        if USE_WANDB:

            log_dict = {"avg_trn_loss": avg_trn_loss, "avg_val_loss": avg_val_loss}

            if epoch == 0:  # one-time logs
                grid = torchv.utils.make_grid(
                    tst_target, normalize=True, nrow=10, pad_value=1, range=(0, 1)
                )
                ex_target = wandb.Image(grid)
                log_dict["Target"] = ex_target

                grid = torchv.utils.make_grid(
                    features, normalize=True, nrow=10, pad_value=1
                )
                km_kernels = wandb.Image(grid)
                log_dict["Kmeans_kernels"] = km_kernels

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

                kernels = model.convT1.weight
                grid = torchv.utils.make_grid(
                    kernels, normalize=True, nrow=10, pad_value=1
                )
                kernels = wandb.Image(grid)
                log_dict["CNN_kernels1"] = kernels

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
