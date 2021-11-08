import numpy as np
from util.whatwhere.encoder import *
from util.whatwhere.decoder import *
from util.willshaw.memory import *
from util.mnist.tools import *
from util.willshaw.plot import *
from util.pickleInterface import *
from util.pytorch.tools import np_to_grid
import wandb
from util.kldiv import *
from tqdm import trange

rng = np.random.RandomState(0)  # reproducible
K = 20  # number of k-means centroids
Q = 21  # size of the final object space grid
n_epochs = 5  # for the k means feature detection
b = 0.8  # minimum activity of the filters: prevents empty feature detection
wta = True  # winner takes all

list_Fs = [1,2,3]  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
list_Tw = [0.5, 0.7, 0.9]  # Treshod for keeping or discarding a detected feature

data_step = 10000
data_max = 60000
save_ret = False


trn_imgs, trn_lbls, tst_imgs, tst_lbls = read_mnist(n_train=60000)

n_runs = len(list_Fs) * len(list_Tw) * (data_max / data_step)
run_idx = 0

if len(sys.argv) < 2:
    print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
    exit(1)
USE_WANDB = bool(int(sys.argv[1]))

for Fs in list_Fs:
    for T_what in list_Tw:
        features = compute_features(trn_imgs, K, Fs, rng, n_epochs, b)

        codes, _ = compute_codes(
            trn_imgs, tst_imgs, K, Q, features, T_what, wta, n_epochs, b, Fs, test=False
        )

        coded_AS, coded_densest = measure_sparsity(codes)

        coded_dist_d, coded_dist_kl, coded_dist_e = measure_data_distribution_set(
            codes.toarray()
        )

        if USE_WANDB:
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
                    "data_step": data_step,
                    "data_max": data_max,
                },
            )

            # 10 examples for visualization purposes
            ex_idxs = idxs_1_random_per_class(trn_lbls[:data_step])
            ex_img = trn_imgs[ex_idxs]
            ex_cod = codes[ex_idxs].toarray()

            # 1-time log
            log_dict = {}
            log_dict["MNIST"] = wandb.Image(np_to_grid(ex_img))
            log_dict["Coded Set"] = wandb.Image(code_grid(ex_cod, K, Q))
            ex_rec_c = reconstruct_set(ex_cod, features, Q, K)  # from coded
            log_dict["Reconstruction (C)"] = wandb.Image(np_to_grid(ex_rec_c))
            wandb.log(log_dict, step=0)

        will = None
        for num_stored in trange(data_step, data_max + 1, data_step, desc="Storing"):

            new_data = codes[num_stored - data_step : num_stored]
            will = incremental_train(new_data, will)
            ret = retreive(codes[:num_stored], will)

            # measurements
            ret_AS, ret_densest = measure_sparsity(ret)
            coded_dist_d, coded_dist_kl, coded_dist_e = measure_data_distribution_set(
                codes[:num_stored].toarray()
            )
            ret_dist_d, ret_dist_kl, ret_dist_e = measure_data_distribution_set(
                ret.toarray()
            )
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
                    "ret_dist_d": ret_dist_d,
                    "ret_dist_kl": ret_dist_kl,
                    "ret_dist_e": ret_dist_e,
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
                }

                ex_ret = ret[ex_idxs].toarray()
                ex_rec_d = reconstruct_set(ex_ret, features, Q, K)  # from ret

                log_dict["Retrieved Set"] = wandb.Image(code_grid(ex_ret, K, Q))
                log_dict["Reconstruction (R)"] = wandb.Image(np_to_grid(ex_rec_d))

                wandb.log(log_dict, step=num_stored)

        if save_ret:
            save_ret(ret, K, Q, Fs, n_epochs, b, T_what)

        if USE_WANDB:
            wandb.finish()
