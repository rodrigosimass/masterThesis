import numpy as np
from util.whatwhere.encoder import *
from util.willshaw.memory import *
from util.willshaw.classifier import *
from util.mnist.tools import *
from util.willshaw.plot import *
from util.pickleInterface import *
import wandb
from util.kldiv import *

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
        features = compute_features(trn_imgs, K, Fs, rng, n_epochs, b)

        codes, _, coded_AS, coded_densest = compute_codes(
            trn_imgs, tst_imgs, K, Q, features, T_what, wta, n_epochs, b, Fs, test=True
        )
        run_name = (
            "k"
            + str(K)
            + "_Fs"
            + str(Fs)
            + "_ep"
            + str(n_epochs)
            + "_b"
            + str(b)
            + "_Q"
            + str(Q)
            + "_Tw"
            + str(T_what)
        )
        print(run_name)
        ret = load_ret(run_name)
        print("codes:")
        code_kl = kl_div_set(codes.toarray(), verbose=True)
        print("ret:")
        ret_kl = kl_div_set(ret.toarray(), verbose=True)

        if use_wandb:
            wandb.init(
                project="whatwhere_kldiv",
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
                    "codes_KL": code_kl,
                    "ret_KL": ret_kl,
                },
            )
        if use_wandb:
            wandb.finish()