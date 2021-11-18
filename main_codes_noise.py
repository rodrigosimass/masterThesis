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

if len(sys.argv) < 2:
    print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
    exit(1)
USE_WANDB = bool(int(sys.argv[1]))

""" PARAMETERS """
rng = np.random.RandomState(0)  # reproducible
K = 20
Q = 21
n_epochs = 5
b = 0.8
wta = True

Fs = 3
T_what = 0.85

list_Pdel = [0.1, 0.2, 0.3, 0.4]
list_Padd = [0.00005, 0.0001, 0.0005, 0.001]

noise_n = 10000

""" load mnist """
trn_imgs, trn_lbls, _, _ = read_mnist(n_train=60000)
I = trn_imgs.shape[1]
J = trn_imgs.shape[2]

""" compute features """
features = compute_features(trn_imgs, trn_lbls, K, Fs, rng, n_epochs, b, verbose=False)

""" generate codes (trn) """
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

code_size = codes.shape[1]  # might not be 60k (some patterns are discarded)

""" each level of noise is a separate wandb run """
for Pdel, Padd in zip(list_Pdel, list_Padd):

    codes_salt = add_zero_noise(codes[:noise_n], prob=Pdel)
    codes_pepper = add_one_noise(codes[:noise_n], prob=Padd)

    if USE_WANDB:

        coded_AS, coded_densest = measure_sparsity(codes)
        coded_shannon_e = shannon_entropy_set(codes.toarray())

        wandb.init(
            project="whatwhere_noise",
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
                "codes_dist_e": coded_shannon_e,
                "noise_Pdel": Pdel,
                "noise_Padd": Padd,
                "noise_n": noise_n,
            },
        )
        wandb.run.name = "noise_" + "Pdel" + str(Pdel) + "Padd" + str(Padd)

        print(get_codes_run_name(K, Fs, n_epochs, b, Q, T_what, wta))

        # examples for visualization purposes (trn set)
        ex_idxs = idxs_x_random_per_class(trn_lbls[:code_size], x=3)
        ex_img = trn_imgs[ex_idxs]
        ex_codes = codes[ex_idxs].toarray()
        ex_polar_params = polar_params[ex_idxs]
        ex_codes_salt = codes_salt[ex_idxs].toarray()
        ex_codes_pepper = codes_pepper[ex_idxs].toarray()

        """ reconstruction without memory """
        ex_recon = recon_img_space(ex_codes, features, ex_polar_params, Q, K, I, J)
        ex_recon_salt = recon_img_space(
            ex_codes_salt, features, ex_polar_params, Q, K, I, J
        )
        ex_recon_pepper = recon_img_space(
            ex_codes_pepper, features, ex_polar_params, Q, K, I, J
        )

        # 1-time log
        log_dict = {}
        log_dict["MNIST"] = wandb.Image(np_to_grid(ex_img))
        log_dict["Coded Set"] = wandb.Image(code_grid(ex_codes, K, Q))
        log_dict["Coded Set (Salt)"] = wandb.Image(code_grid(ex_codes_salt, K, Q))
        log_dict["Coded Set (Pepper)"] = wandb.Image(code_grid(ex_codes_pepper, K, Q))
        log_dict["Reconstruction"] = wandb.Image(np_to_grid(ex_recon))
        log_dict["Reconstruction (Salt)"] = wandb.Image(np_to_grid(ex_recon_salt))
        log_dict["Reconstruction (Pepper)"] = wandb.Image(np_to_grid(ex_recon_pepper))
        wandb.log(log_dict, step=0)

    max_fos = int(codes.shape[0] / codes.shape[1])

    will = None
    for fos in trange(max_fos, desc="Storing", unit="factor of stored (fos)"):

        num_stored = code_size * (fos + 1)

        new_data = codes[num_stored - code_size : num_stored]
        will = incremental_train(new_data, will)

        ret = retreive(codes[:num_stored], will)
        ret_salt = retreive(codes_salt[:noise_n], will)
        ret_pepper = retreive(codes_pepper[:noise_n], will)

        """ measure sparseness and distribution """
        ret_AS, ret_densest = measure_sparsity(ret)
        ret_salt_AS, ret_salt_densest = measure_sparsity(ret_salt)
        ret_pepper_AS, ret_pepper_densest = measure_sparsity(ret_pepper)
        will_S = willshaw_sparsity(will)

        ret_shannon_e = shannon_entropy_set(ret.toarray())
        ret_salt_shannon_e = shannon_entropy_set(ret_salt.toarray())
        ret_pepper_shannon_e = shannon_entropy_set(ret_pepper.toarray())

        """ create reconstructions """
        recons = recon_img_space(
            ret.toarray(), features, polar_params[:num_stored], Q, K, I, J
        )
        recons_salt = recon_img_space(
            ret_salt.toarray(), features, polar_params[:noise_n], Q, K, I, J
        )
        recons_pepper = recon_img_space(
            ret_pepper.toarray(), features, polar_params[:noise_n], Q, K, I, J
        )

        """ measure reconstruction MSE """
        mse_recon = mse(trn_imgs[:num_stored].flatten(), recons.flatten())
        mse_recon_salt = mse(trn_imgs[:noise_n].flatten(), recons_salt.flatten())
        mse_recon_pepper = mse(trn_imgs[:noise_n].flatten(), recons_pepper.flatten())

        """ measure preformance """
        err_perfRet, err_avgErr, err_loss, err_noise, err_1nn = performance(
            codes, ret, trn_lbls, verbose=False
        )
        (
            err_salt_perfRet,
            err_salt_avgErr,
            err_salt_loss,
            err_salt_noise,
            err_salt_1nn,
        ) = performance(codes_salt, ret_salt, trn_lbls, verbose=False)
        (
            err_pepper_perfRet,
            err_pepper_avgErr,
            err_pepper_loss,
            err_pepper_noise,
            err_pepper_1nn,
        ) = performance(codes_pepper, ret_pepper, trn_lbls, verbose=False)

        if USE_WANDB:
            # step-wise log
            log_dict = {
                "cod_AS": coded_AS,  # for horizontal line ref
                "cod_%B": coded_densest,  # for horizontal line ref
                "will_S": will_S,
                "ret_AS": ret_AS,
                "ret_densest": ret_densest,
                "ret_salt_AS": ret_salt_AS,
                "ret_salt_densest": ret_salt_densest,
                "ret_pepper_AS": ret_pepper_AS,
                "ret_pepper_densest": ret_pepper_densest,
                "ret_shannon_e": ret_shannon_e,
                "ret_salt_shannon_e": ret_salt_shannon_e,
                "ret_pepper_shannon_e": ret_pepper_shannon_e,
                "err_1NN": err_1nn,
                "err_perfRet": err_perfRet,
                "err_avgErr": err_avgErr,
                "err_loss": err_loss,
                "err_noise": err_noise,
                "err_salt_1NN": err_salt_1nn,
                "err_salt_perfRet": err_salt_perfRet,
                "err_salt_avgErr": err_salt_avgErr,
                "err_salt_loss": err_salt_loss,
                "err_salt_noise": err_salt_noise,
                "err_pepper_1NN": err_pepper_1nn,
                "err_pepper_perfRet": err_pepper_perfRet,
                "err_pepper_avgErr": err_pepper_avgErr,
                "err_pepper_loss": err_pepper_loss,
                "err_pepper_noise": err_pepper_noise,
                "mse_recon": mse_recon,
                "mse_recon_salt": mse_recon_salt,
                "mse_recon_pepper": mse_recon_pepper,
            }

            ex_ret = ret[ex_idxs].toarray()
            ex_ret_salt = ret_salt[ex_idxs].toarray()
            ex_ret_pepper = ret_pepper[ex_idxs].toarray()

            ex_recon = recon_img_space(ex_ret, features, ex_polar_params, Q, K, I, J)
            ex_recon_salt = recon_img_space(
                ex_ret_salt, features, ex_polar_params, Q, K, I, J
            )
            ex_recon_pepper = recon_img_space(
                ex_ret_pepper, features, ex_polar_params, Q, K, I, J
            )

            log_dict["Retrieved Set"] = wandb.Image(code_grid(ex_ret, K, Q))
            log_dict["Retrieved Set (Salt)"] = wandb.Image(code_grid(ex_ret_salt, K, Q))
            log_dict["Retrieved Set (Pepper)"] = wandb.Image(
                code_grid(ex_ret_pepper, K, Q)
            )
            log_dict["Reconstruction"] = wandb.Image(np_to_grid(ex_recon))
            log_dict["Reconstruction (Salt)"] = wandb.Image(np_to_grid(ex_recon_salt))
            log_dict["Reconstruction (Pepper)"] = wandb.Image(
                np_to_grid(ex_recon_pepper)
            )

            wandb.log(log_dict, step=num_stored)

    if USE_WANDB:
        wandb.finish()
