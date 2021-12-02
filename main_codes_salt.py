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
list_Pdel = [0, 0.05, 0.1, 0.15]  # each item in this list is a different wandb run

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


for Pdel in list_Pdel:

    codes_salt = add_zero_noise(codes, prob=Pdel)

    codes_AS = measure_sparsity(codes)[0]
    codes_salt_AS = measure_sparsity(codes_salt)[0]

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
                "noise_Pdel": Pdel,
                "trial": trial_run,
            },
        )

        name = "TRIAL_" if trial_run else ""
        wandb.run.name = name + "neg_pos_mse" + "Pdel" + str(Pdel)
        print(get_codes_run_name(K, Fs, n_epochs, b, Q, T_what, wta))

        """ log the initial state (no memory) """
        log_dict = {}
        # log_dict["codes_AS"] = codes_AS
        # log_dict["codes_salt_AS"] = codes_salt_AS

        # reconstructions
        recons = recon_img_space(codes, features, polar_params, Q, K, I, J)
        nse, pse, mse = mse_detailed(recons, imgs)
        recons_salt = recon_img_space(codes_salt, features, polar_params, Q, K, I, J)
        nse_salt, pse_salt, mse_salt = mse_detailed(recons_salt, imgs)
        log_dict["mse"] = mse
        log_dict["nse"] = nse
        log_dict["pse"] = pse
        log_dict["mse_salt"] = mse_salt
        log_dict["nse_salt"] = nse_salt
        log_dict["pse_salt"] = pse_salt

        """ examples for visualization purposes """
        ex_idxs = idxs_x_random_per_class(lbls[:code_size], x=3)
        ex_img = imgs[ex_idxs]
        ex_codes = codes[ex_idxs]
        ex_polar_params = polar_params[ex_idxs]
        ex_codes_salt = codes_salt[ex_idxs]
        ex_recon = recons[ex_idxs]
        ex_recon_salt = recons_salt[ex_idxs]

        log_dict["MNIST"] = wandb.Image(np_to_grid(ex_img))
        log_dict["Codes"] = wandb.Image(code_grid(ex_codes, K, Q))
        log_dict["Codes (Salt)"] = wandb.Image(code_grid(ex_codes_salt, K, Q))
        log_dict["Recon"] = wandb.Image(np_to_grid(ex_recon))
        log_dict["Recon (Salt)"] = wandb.Image(np_to_grid(ex_recon_salt))

        wandb.log(log_dict, step=0)

    max_fos = int(codes.shape[0] / codes.shape[1])

    will = None  # empty willshaw matrix
    for fos in trange(max_fos, desc="Storing", unit="factor of stored (fos)"):

        num_stored = code_size * (fos + 1)

        new_data = codes[num_stored - code_size : num_stored]
        will = incremental_train(new_data, will)

        ret = retreive(codes[:num_stored], will)
        ret_salt = retreive(codes_salt, will)

        """ create reconstructions """
        recons = recon_img_space(ret, features, polar_params[:num_stored], Q, K, I, J)
        recons_salt = recon_img_space(ret_salt, features, polar_params, Q, K, I, J)

        """ measure reconstruction MSE """
        nse, pse, mse = mse_detailed(imgs[:num_stored], recons)
        nse_salt, pse_salt, mse_salt = mse_detailed(imgs, recons_salt)

        """ measure preformance """
        err_perfRet, err_avgErr, err_loss, err_noise, err_1nn = performance(
            codes, ret, lbls, verbose=False
        )
        (
            err_salt_perfRet,
            err_salt_avgErr,
            err_salt_loss,
            err_salt_noise,
            err_salt_1nn,
        ) = performance(codes_salt, ret_salt, lbls, verbose=False)

        if USE_WANDB:
            """log metrics"""
            log_dict = {
                "err_1NN": err_1nn,
                "err_avgErr": err_avgErr,
                "err_salt_1NN": err_salt_1nn,
                "err_salt_avgErr": err_salt_avgErr,
                "mse": mse,
                "nse": nse,
                "pse": pse,
                "mse_salt": mse_salt,
                "nse_salt": nse_salt,
                "pse_salt": pse_salt,
            }

            """ log images """
            log_dict["Ret"] = wandb.Image(code_grid(ret[ex_idxs], K, Q))
            log_dict["Ret (Salt)"] = wandb.Image(code_grid(ret_salt[ex_idxs], K, Q))
            log_dict["Recon"] = wandb.Image(np_to_grid(recons[ex_idxs]))
            log_dict["Recon (Salt)"] = wandb.Image(np_to_grid(recons_salt[ex_idxs]))

            wandb.log(log_dict, step=num_stored)

    if USE_WANDB:
        wandb.finish()
