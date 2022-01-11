#%%
import numpy as np
import wandb
from tqdm import trange
from util.mnist.tools import *
from util.pickleInterface import *
from util.whatwhere.encoder import *
from util.whatwhere.decoder import *
from util.whatwhere.noise import *
from util.willshaw.memory import *
from util.willshaw.plot import *
from util.pytorch.tools import np_to_grid
from util.kldiv import *
from util.basic_utils import mse_detailed

#%%

if len(sys.argv) < 2:
    print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
    exit(1)
USE_WANDB = bool(int(sys.argv[1]))

#%%

""" Code generation parameters """
rng = np.random.RandomState(0)  # reproducible
K = 20
Q = 21
n_epochs = 5
b = 0.8
wta = True
Fs = 2
T_what = 0.95

trial_run = False
l_prob = [0.1, 0.2]  # each item in this list is a different wandb run
noise_type = "one"  # zero, one or none

""" load mnist """
imgs, lbls, _, _ = read_mnist(n_train=60000)
I = imgs.shape[1]
J = imgs.shape[2]


""" generate codes """
features = compute_features(imgs, lbls, K, Fs, rng, n_epochs, b, verbose=False)
codes, polar_params = compute_codes(
    imgs,
    K,
    Q,
    features,
    T_what,
    wta,
    n_epochs,
    b,
    Fs,
    verbose=False,
)

#%%

code_size = codes.shape[1]
if trial_run:
    # Reduce the size of the datasets for debugging
    imgs = imgs[: code_size * 2]
    lbls = lbls[: code_size * 2]
    lbls = lbls[: code_size * 2]
    codes = codes[: code_size * 2]
    polar_params = polar_params[: code_size * 2]

for prob in l_prob:
    if noise_type == "zero":
        codes_noisy = add_zero_noise(codes, prob)
    elif noise_type == "one":
        codes_noisy = add_one_noise_relative(codes, prob)
    elif noise_type == "none":
        codes_noisy = codes
    else:
        print(
            f"Unknown noise type: <<{noise_type}>>.\nUse: <<one>>, <<zero>>, or <<none>>"
        )

    if USE_WANDB:
        wandb.init(
            project="whatwhere_salt",
            entity="rodrigosimass",
            config={
                "km_K": K,
                "km_epochs": n_epochs,
                "km_wta": wta,
                "km_b": b,
                "km_Fs": Fs,
                "ww_Q": Q,
                "ww_Twhat": T_what,
                "noise_type": noise_type,
                "noise_prob": prob,
            },
        )

        name = "TRIAL_" if trial_run else ""
        name += noise_type
        if noise_type != "none":
            name += "_p" + str(prob)
        wandb.run.name = name

        """ log the initial state (no memory) """
        log_dict = {
            "err_pre": 0.0,
            "err_hd_extra": 0.0,
            "err_hd_lost": 0.0,
            "err_hd": 0.0,
            "err_err_1nn": 0.0,
        }

        sparsity = measure_sparsity(codes_noisy)
        log_dict["sparsity_average"] = sparsity[0]
        log_dict["sparsity_densest"] = sparsity[1]

        # reconstructions
        recons = recon_with_polar(codes_noisy, features, polar_params, Q, K)
        extra, lost, mse = mse_detailed(recons, imgs)

        log_dict["mse_extra"] = extra
        log_dict["mse_lost"] = lost
        log_dict["mse"] = mse

        """ examples for visualization purposes """
        idx = idxs_x_random_per_class(lbls[:code_size], x=3, seed=True)
        ex_img = imgs[idx]
        ex_codes_noisy = codes_noisy[idx]
        ex_polar_params = polar_params[idx]
        ex_recon = recons[idx]

        log_dict["MNIST"] = wandb.Image(np_to_grid(ex_img))
        log_dict["Ret"] = wandb.Image(code_grid(ex_codes_noisy, K, Q))
        log_dict["Recon"] = wandb.Image(np_to_grid(ex_recon))

        wandb.log(log_dict, step=0)

    max_fos = int(codes.shape[0] / codes.shape[1])

    wn = AAWN(code_size)  # empty memory
    for fos in trange(max_fos, desc="Storing", unit="factor of stored (fos)"):

        num_stored = code_size * (fos + 1)

        """ train the memory with clean data """
        new_data = codes[num_stored - code_size : num_stored]
        wn.store(new_data)

        """ present the memory with noisy versions of the data """
        ret = wn.retrieve(codes_noisy[:num_stored])

        """ create reconstructions """
        recons = recon_with_polar(ret, features, polar_params[:num_stored], Q, K)

        """ measure reconstruction MSE """
        extra, lost, mse = mse_detailed(recons, imgs[:num_stored])

        """ evaluate quality of codes """
        err = eval(
            codes[:num_stored], lbls[:num_stored], ret[:num_stored], lbls[:num_stored]
        )

        if USE_WANDB:
            log_dict = {
                "err_pre": err[0],
                "err_hd_extra": err[1],
                "err_hd_lost": err[2],
                "err_hd": err[3],
                "err_err_1nn": err[4],
                "mse_extra": extra,
                "mse_lost": lost,
                "mse": mse,
            }
            sparsity = measure_sparsity(ret)
            log_dict["sparsity_average"] = sparsity[0]
            log_dict["sparsity_densest"] = sparsity[1]

            """ log images """
            log_dict["Ret"] = wandb.Image(code_grid(ret[idx], K, Q))
            log_dict["Recon"] = wandb.Image(np_to_grid(recons[idx]))

            wandb.log(log_dict, step=num_stored)

    if USE_WANDB:
        wandb.finish()
