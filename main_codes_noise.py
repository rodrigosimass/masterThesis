import numpy as np
import wandb
from tqdm import trange
from sklearn.metrics import mean_squared_error
from util.mnist.tools import *
from util.pickleInterface import *
from util.whatwhere.encoder import *
from util.whatwhere.decoder import *
from util.whatwhere.noise import add_one_noise, add_zero_noise
from util.willshaw.memory import *
from util.willshaw.plot import *
from util.pytorch.tools import np_to_grid
from util.kldiv import *
from util.basic_utils import mse

rng = np.random.RandomState(0)  # reproducible
K = 30
Q = 10
n_epochs = 5
b = 0.8
wta = True

Fs = 3
T_what = 0.85

""" list_Pdel = [0, 0.05, 0.1, 0.15]
list_Padd = [0, 0.0001, 0.005, 0.01] """
list_Pdel = [0.15]
list_Padd = [0.01]


trn_imgs, trn_lbls, tst_imgs, tst_lbls = read_mnist(n_train=60000)
I = trn_imgs.shape[1]
J = trn_imgs.shape[2]

if len(sys.argv) < 2:
    print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
    exit(1)
USE_WANDB = bool(int(sys.argv[1]))


for Pdel, Padd in zip(list_Pdel, list_Padd):
    features = compute_features(
        trn_imgs, trn_lbls, K, Fs, rng, n_epochs, b, verbose=False
    )

    codes, polar_params = compute_codes(
        trn_imgs,
        K,
        Q,
        features,
        T_what,
        wta,
        n_epochs,
        b,
        Fs,
        verbose=False,
        set="trn",
    )

    codes_salt = add_zero_noise(codes, prob=Pdel)
    codes_pepper = add_one_noise(codes, prob=Padd)

    code_size = codes.shape[1]  # might not be 60k (some patterns are discarded)

    if USE_WANDB:

        coded_AS, coded_densest = measure_sparsity(codes)

        (
            coded_dist_d,
            coded_dist_kl,
            coded_dist_e,
        ) = measure_data_distribution_set(codes.toarray())

        wandb.init(
            project="whatwhere_mock",
            entity="rodrigosimass",
            config={
                "km_K": K,
                "km_epochs": n_epochs,
                "km_wta": wta,
                "km_b": b,
                "km_Fs": Fs,
                "ww_Q": Q,
                "ww_Twhat": T_what,
                "codes_AS": coded_AS,
                "codes_%B": coded_densest,
                "codes_dist_d": coded_dist_d,
                "codes_dist_kl": coded_dist_kl,
                "codes_dist_e": coded_dist_e,
                "noise_Pdel": Pdel,
                "noise_Padd": Padd,
            },
        )
        wandb.run.name = (
            "noise_"
            + "Fs"
            + str(Fs)
            + "Tw"
            + str(T_what)
            + "Pdel"
            + str(Pdel)
            + "Padd"
            + str(Padd)
        )

        # examples for visualization purposes
        ex_idxs = idxs_x_random_per_class(trn_lbls[:code_size], x=3)
        ex_img = trn_imgs[ex_idxs]
        ex_codes = codes[ex_idxs].toarray()
        ex_polar_params = polar_params[ex_idxs]
        ex_codes_salt = codes_salt[ex_idxs].toarray()
        ex_codes_pepper = codes_pepper[ex_idxs].toarray()

        ex_recon = recon_img_space(
            ex_codes, features, polar_params[ex_idxs], Q, K, I, J
        )

        # 1-time log
        log_dict = {}
        log_dict["MNIST"] = wandb.Image(np_to_grid(ex_img))
        log_dict["Coded Set"] = wandb.Image(code_grid(ex_codes, K, Q))
        log_dict["Coded Set (Salt)"] = wandb.Image(code_grid(ex_codes_salt, K, Q))
        log_dict["Coded Set (Pepper)"] = wandb.Image(code_grid(ex_codes_pepper, K, Q))
        log_dict["Reconstruction"] = wandb.Image(np_to_grid(ex_rec_img))
        wandb.log(log_dict, step=0)

    max_fos = int(codes.shape[0] / codes.shape[1])
    max_fos = 2

    will = None
    for fos in trange(max_fos, desc="Storing", unit="factor of stored (fos)"):

        num_stored = code_size * (fos + 1)

        new_data = codes[num_stored - code_size : num_stored]
        will = incremental_train(new_data, will)
        ret = retreive(codes[:num_stored], will)
        ret_salt = retreive(codes_salt[:num_stored], will)
        ret_pepper = retreive(codes_pepper[:num_stored], will)

        recons = recon_img_space(
            ret.toarray(), features, polar_params[:num_stored], Q, K, I, J
        )

        mse_recon = mse(trn_imgs[:num_stored].flatten(), recons.flatten())

        recons_salt = recon_img_space(
            ret_salt.toarray(), features, polar_params[:num_stored], Q, K, I, J
        )

        mse_recon_salt = mse(trn_imgs[:num_stored].flatten(), recons_salt.flatten())

        recons_pepper = recon_img_space(
            ret_pepper.toarray(), features, polar_params[:num_stored], Q, K, I, J
        )

        mse_recon_pepper = mse(trn_imgs[:num_stored].flatten(), recons_pepper.flatten())

        # measurements
        ret_AS, ret_densest = measure_sparsity(ret)
        ret_salt_AS, ret_salt_densest = measure_sparsity(ret_salt)
        ret_pepper_AS, ret_pepper_densest = measure_sparsity(ret_pepper)
        will_S = willshaw_sparsity(will)
        err_perfRet, err_avgErr, err_infoLoss, err_noise, err_1nn = performance(
            codes, ret, trn_lbls, verbose=False
        )

        if USE_WANDB:
            # step-wise log
            log_dict = {
                "will_S": will_S,
                "ret_AS": ret_AS,
                "ret_%B": ret_densest,
                "cod_AS": coded_AS,
                "cod_%B": coded_densest,
                "cod_dist_d": coded_dist_d,
                "cod_dist_kl": coded_dist_kl,
                "cod_dist_e": coded_dist_e,
                "err_1NN": err_1nn,
                "err_perfRet": err_perfRet,
                "err_infoLoss": err_infoLoss,
                "err_noise": err_noise,
                "err_avgErr": err_avgErr,
                "mse_recon": mse_recon,
                "mse_recon_salt": mse_recon_salt,
                "mse_recon_pepper": mse_recon_pepper,
            }

            ex_ret = ret[ex_idxs].toarray()
            ex_ret_salt = ret_salt[ex_idxs].toarray()
            ex_ret_pepper = ret_pepper[ex_idxs].toarray()

            ex_recon = recon_img_space(
                ex_ret, features, polar_params[ex_idxs], Q, K, I, J
            )
            ex_recon_salt = recon_img_space(
                ex_ret_salt, features, polar_params[ex_idxs], Q, K, I, J
            )
            ex_recon_pepper = recon_img_space(
                ex_ret_pepper, features, polar_params[ex_idxs], Q, K, I, J
            )

            log_dict["Retrieved Set"] = wandb.Image(code_grid(ex_ret, K, Q))
            log_dict["Reconstruction"] = wandb.Image(np_to_grid(ex_recon))
            log_dict["Reconstruction (Salt)"] = wandb.Image(np_to_grid(ex_recon_salt))
            log_dict["Reconstruction (Pepper)"] = wandb.Image(
                np_to_grid(ex_recon_pepper)
            )

            wandb.log(log_dict, step=num_stored)

        if USE_WANDB:
            wandb.finish()
