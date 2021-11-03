import wandb
import sys
import torch
import torchvision as torchv
from torch.utils.data import TensorDataset, DataLoader, random_split
from util.pickleInterface import load_codes, get_codes_run_name, get_features_run_name, load_features
from util.mnist.tools import *
from util.pytorch.earlystopping import *
from util.pytorch.print_log import print_model_info
from models.deCNN import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
        exit(1)
    USE_WANDB = bool(int(sys.argv[1]))
    
    """ ---------- PARAMS ---------- """
    # CODES
    k = 20
    Fs = 2
    n_epochs = 5
    b = 0.8
    Q = 21
    T_what = 0.95
    code_id = get_codes_run_name(k, Fs, n_epochs, b, Q, T_what)

    # UPSCALE LAYER
    l1_in_dim = k * Q * Q  # Codes after retinotopic step
    l1_out_dim = k * 28 * 28  # codes before retinotopic step

    # early stopping
    patience = 10
    delta = 1e-4

    # train
    max_epochs = 100
    lr = 3e-3

    # Dataset sizes
    batch_size = 8
    size = 2000
    
    trn_n = 6 * size
    val_n = 2 * size  
    tst_n = 1 * size

    tst_model = True
    save_model = True

    # CNN params are computes from the WW encoder params
    init_kernels = False
    convT_in_ch = k
    convT_out_ch = 1
    convT_k_size = 2 * Fs + 1
    convT_pad = (Fs, Fs)  # for same padding

    model_name = "CNN" +  "_init_" + str(init_kernels) +  "_n_"  + str(trn_n)
    
    trn_imgs, _, tst_imgs, tst_lbls = read_mnist()
    trn_imgs = trn_imgs[:trn_n].reshape((trn_n, 28 * 28))
    tst_imgs = tst_imgs[:tst_n].reshape((tst_n, 28 * 28))
    tst_lbls = tst_lbls[:tst_n]
    
    trn_codes, tst_codes = load_codes(code_id)
    trn_codes = trn_codes[:trn_n].toarray().reshape((-1, k, Q, Q))
    tst_codes = tst_codes[:tst_n].toarray().reshape((-1, k, Q, Q))

    # create a tensor of the kernels that generated the codes
    features = load_features(get_features_run_name(k,Fs, n_epochs, b))
    features = torch.from_numpy(features)
    features = torch.unsqueeze(features, dim=1)

    # create a tensor from the test set for each class (to visualize reconstructions) 
    idxs = idxs_1_random_per_class(tst_lbls)
    tst_in = torch.from_numpy(tst_codes[idxs])
    tst_target = torch.from_numpy(tst_imgs[idxs])
    tst_target = tst_target.reshape((-1, 28, 28)) 
    tst_target = torch.unsqueeze(tst_target, dim=1) # Add the channel dimention

    trn_dataset = TensorDataset(torch.Tensor(trn_codes), torch.Tensor(trn_imgs))
    tst_dataset = TensorDataset(torch.Tensor(tst_codes), torch.Tensor(tst_imgs))

    trn_dataset, val_dataset = random_split(trn_dataset, [trn_n - val_n, val_n])

    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    tst_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Selected device: {device}')

    model = deCNN_MLP(
            l1_in_dim,
            l1_out_dim,
            convT_in_ch,
            convT_out_ch,
            convT_k_size,
            convT_pad).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_function = nn.MSELoss()

    #TODO: maybe init with Kmeans kernels
    if init_kernels:
        model.state_dict()["convT1.weight"][:] = features


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
                "ww_Tw": T_what,
                "convT_in_ch": convT_in_ch,
                "convT_out_ch": convT_out_ch,
                "convT_k_size": convT_k_size,
                "convT_pad": convT_pad,
                "init_kernels": init_kernels,
                "ann_lr": lr,
                "ann_max_epochs": max_epochs,
                "ann_batch_size": batch_size,
                "ann_patience": patience,
                "ann_trn_n": trn_n,
                "ann_val_n": val_n,
                "ann_loss": str(loss_function),
            },
        )

    print_model_info(model, optimizer, loss_function, verbose=2)

    if patience > 0:
        early_stopping = EarlyStopping(
            patience=patience, verbose=True, delta=delta, name=model_name, save_model=save_model
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
        trn_loss = trn_loss / len(trn_loader) #TODO: chamar isto avg_trn_loss

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

        val_loss = val_loss / len(val_loader)

        #print(f"Loss (train) = {trn_loss} ; Loss (validation) = {val_loss}")

        # Early Stopping
        if patience > 0:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Log stats
        if USE_WANDB:

            log_dict = {"trn_loss":trn_loss, "val_loss": val_loss}

            if epoch == 0: # one-time logs
                grid = torchv.utils.make_grid(tst_target, normalize=True, nrow=10, pad_value=1, range=(0,1))
                ex_target = wandb.Image(grid)
                log_dict["Target"] = ex_target

                grid = torchv.utils.make_grid(features, normalize=True, nrow=10, pad_value=1)
                km_kernels = wandb.Image(grid)
                log_dict["Kmeans_kernels"] = km_kernels

            if early_stopping.counter == 0 or epoch == 0: #Improved -> log reconstructions
                with torch.no_grad():
                    tst_in = tst_in.to(device)
                    tst_out = model(tst_in.float())
                tst_out = tst_out.reshape((-1, 28, 28))
                tst_out = torch.unsqueeze(tst_out, dim=1) # Add empty channel dimention
                grid = torchv.utils.make_grid(tst_out, normalize=True, nrow=10, pad_value=1, range=(0,1))
                ex_out = wandb.Image(grid)
                log_dict["Output"] = ex_out

                kernels = model.convT1.weight
                grid = torchv.utils.make_grid(kernels, normalize=True, nrow=10, pad_value=1)
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
            wandb.log({"tst_loss":tst_loss})

        print(f"Test set avg. MSE loss = {tst_loss}")

    
    if USE_WANDB:
        wandb.finish()