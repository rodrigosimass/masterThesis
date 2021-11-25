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


""" load mnist """
trn_imgs, trn_lbls, tst_imgs, tst_lbls = read_mnist(n_train=60000)
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

""" generate codes (tst) """
tst_codes, tst_polar_params = compute_codes(
    tst_imgs,
    K,
    Q,
    features,
    T_what,
    wta,
    n_epochs,
    b,
    Fs,
    verbose=False,
    set="tst",
)

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
        },
    )
    wandb.run.name = "noise_" + "Fs" + str(Fs) + "Tw" + str(T_what) + "Pdel"

    print(get_codes_run_name(K, Fs, n_epochs, b, Q, T_what, wta))

    # examples for visualization purposes (trn set)
    ex_idxs = idxs_x_random_per_class(trn_lbls[:code_size], x=3)
    ex_img = trn_imgs[ex_idxs]
    ex_codes = codes[ex_idxs]
    ex_polar_params = polar_params[ex_idxs]

    # examples for visualization purposes (tst set)
    tst_ex_idxs = idxs_x_random_per_class(tst_lbls[: tst_codes.shape[0]], x=3)
    tst_ex_img = tst_imgs[tst_ex_idxs]
    tst_ex_codes = tst_codes[tst_ex_idxs]
    tst_ex_polar_params = tst_polar_params[tst_ex_idxs]

    """ reconstruction without memory """
    ex_recon = recon_img_space(ex_codes, features, ex_polar_params, Q, K, I, J)
    tst_ex_recon = recon_img_space(
        tst_ex_codes, features, tst_ex_polar_params, Q, K, I, J
    )

    # 1-time log
    log_dict = {}
    log_dict["MNIST"] = wandb.Image(np_to_grid(ex_img))
    log_dict["Coded Set"] = wandb.Image(code_grid(ex_codes, K, Q))
    log_dict["Reconstruction"] = wandb.Image(np_to_grid(ex_recon))

    log_dict["MNIST (tst)"] = wandb.Image(np_to_grid(tst_ex_img))
    log_dict["Coded Set (tst)"] = wandb.Image(code_grid(tst_ex_codes, K, Q))
    log_dict["Reconstruction (tst)"] = wandb.Image(np_to_grid(tst_ex_recon))

    wandb.log(log_dict, step=0)

max_fos = int(codes.shape[0] / codes.shape[1])

will = None
for fos in trange(max_fos, desc="Storing", unit="factor of stored (fos)"):

    num_stored = code_size * (fos + 1)

    new_data = codes[num_stored - code_size : num_stored]
    will = incremental_train(new_data, will)
    ret = retreive(codes[:num_stored], will)

    tst_ret = retreive(tst_codes, will)

    """ create reconstructions """
    recons = recon_img_space(
        ret.toarray(), features, polar_params[:num_stored], Q, K, I, J
    )
    tst_recons = recon_img_space(
        tst_ret.toarray(), features, tst_polar_params, Q, K, I, J
    )

    """ measure reconstruction MSE """
    mse_recon = mse(trn_imgs[:num_stored].flatten(), recons.flatten())
    tst_mse_recon = mse(tst_imgs.flatten(), tst_recons.flatten())

    """ measure sparseness """
    ret_AS, ret_densest = measure_sparsity(ret)
    tst_ret_AS, tst_ret_densest = measure_sparsity(tst_ret)
    will_S = willshaw_sparsity(will)

    """ measure preformance """
    err_perfRet, err_avgErr, _, _, err_1nn = performance(
        codes, ret, trn_lbls, verbose=False
    )
    err_tst_perfRet, err_tst_avgErr, _, _, err_tst_1nn = performance(
        tst_codes, tst_ret, tst_lbls, verbose=False
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
            "err_avgErr": err_avgErr,
            "err_tst_1NN": err_tst_1nn,
            "err_tst_perfRet": err_tst_perfRet,
            "err_tst_avgErr": err_tst_avgErr,
            "mse_recon": mse_recon,
            "tst_mse_recon": tst_mse_recon,
        }

        ex_ret = ret[ex_idxs]
        tst_ex_ret = tst_ret[tst_ex_idxs]

        ex_recon = recon_img_space(ex_ret, features, ex_polar_params, Q, K, I, J)
        tst_ex_recon = recon_img_space(
            tst_ex_ret, features, tst_ex_polar_params, Q, K, I, J
        )

        log_dict["Retrieved Set"] = wandb.Image(code_grid(ex_ret, K, Q))
        log_dict["Retrieved Set (tst)"] = wandb.Image(code_grid(tst_ex_ret, K, Q))
        log_dict["Reconstruction"] = wandb.Image(np_to_grid(ex_recon))
        log_dict["Reconstruction (tst)"] = wandb.Image(np_to_grid(tst_ex_recon))

        wandb.log(log_dict, step=num_stored)

if USE_WANDB:
    wandb.finish()
