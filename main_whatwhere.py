import numpy as np
from util.whatwhere.encoder import *
from util.willshaw.memory import *
from util.willshaw.classifier import *
from util.mnist.loader import *
from util.willshaw.plot import *
from util.pickleInterface import *
import wandb

rng = np.random.RandomState(0)  # reproducible
""" Fixed params """
K = 20  # number of k-means centroids
Q = 21  # size of the final object space grid
n_epochs = 5  # for the k means feature detection
b = 0.8  # minimum activity of the filters: prevents empty feature detection
wta = True  # winner takes all

""" Variable params """
list_Fs = [1, 2, 3]  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
list_Tw = [
    0.8,
    0.85,
    0.9,
    0.95,
]  # Treshod for keeping or discarding a detected feature

data_step = 6000
data_max = 60000

trn_imgs, trn_lbls, tst_imgs, _ = read_mnist(n_train=60000)

n_runs = len(list_Fs) * len(list_Tw) * (data_max / data_step)
run_idx = 0

use_wandb = True

for Fs in list_Fs:
    for T_what in list_Tw:
        features = load_or_compute_features(trn_imgs, K, Fs, rng, n_epochs, b)

        codes, _, coded_AS, coded_densest = load_or_compute_codes(
            trn_imgs, tst_imgs, K, Q, features, T_what, wta, n_epochs, b, Fs, test=True
        )
        if use_wandb:
            wandb.init(
                project="whatwhere",
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
                },
            )
            example_grid = get_codes_examples(codes, K, Q)
            examples = wandb.Image(example_grid, caption="Examples of codes")
            wandb.log({"cod_examples": examples})

        will = None
        for num_stored in range(data_step, data_max + 1, data_step):
            run_idx += 1
            print(f"\nRun {run_idx}/{n_runs}")

            new_data = codes[num_stored - data_step : num_stored]
            will, will_S = incremental_train(new_data, will)

            ret, ret_AS, ret_densest = retreive(codes, num_stored, will)
            err_perfRet, err_avgErr, err_infoLoss, err_noise, err_1nn = performance(
                codes, ret, trn_lbls, verbose=True
            )
            if use_wandb:
                wandb.log(
                    {
                        "will_S": will_S,
                        "ret_AS": ret_AS,
                        "ret_%B": ret_densest,
                        "cod_AS": coded_AS,
                        "cod_%B": coded_densest,
                        "err_1NN": err_1nn,
                        "err_perfRet": err_perfRet,
                        "err_infoLoss": err_infoLoss,
                        "err_noise": err_noise,
                        "err_avgErr": err_avgErr,
                    },
                    step=num_stored,
                )
        store_ret(ret, K, Q, data_max, Fs, n_epochs, b, T_what)
        if use_wandb:
            example_grid = get_codes_examples(ret, K, Q)
            examples = wandb.Image(example_grid, caption="Examples of retrievals")
            wandb.log({"ret_examples": examples})
            wandb.finish()