"""
Reproduction of Sa-Conto and Wichert 2020
"""
import numpy as np
from util.whatwhere.encoder import *
from util.whatwhere.decoder import *
from util.whatwhere.plot import *
from util.willshaw.memory import *
from util.mnist.tools import *
from util.willshaw.plot import *
from util.pickleInterface import *
from util.pytorch.tools import np_to_grid
import wandb
from util.distribution import *
from tqdm import trange

rng = np.random.RandomState(0)  # reproducible
K = 20
Q = 21
n_epochs = 5
b = 0.8
wta = True

list_Fs = [1, 2]
list_Tw = [0.75, 0.8, 0.85]


TRIAL_RUN = False  # if True: Reduce the size of the datasets for debugging

imgs, lbls, _, _ = read_mnist(n_train=60000)

if len(sys.argv) < 2:
    print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
    exit(1)
USE_WANDB = bool(int(sys.argv[1]))

for Tw in list_Tw:
    for Fs in list_Fs:
        features = compute_features(imgs, lbls, K, Fs, rng, n_epochs, b)

        codes, _ = compute_codes(
            imgs,
            K,
            Q,
            features,
            Tw,
            wta,
            n_epochs,
            b,
            Fs,
            set="trn",
        )

        code_size = codes.shape[1]

        if TRIAL_RUN:
            imgs = imgs[: code_size * 2]
            lbls = lbls[: code_size * 2]
            codes = codes[: code_size * 2]

        coded_AS, coded_densest = measure_sparsity(codes)
        codes_e = shannon_entropy_set(codes.toarray())

        if USE_WANDB:

            wandb.init(
                project="main_luis_paper",
                entity="rodrigosimass",
                config={
                    "km_K": K,
                    "km_epochs": n_epochs,
                    "km_wta": wta,
                    "km_b": b,
                    "km_Fs": Fs,
                    "ww_Q": Q,
                    "ww_Twhat": Tw,
                    "codes_AS": coded_AS,
                    "codes_%B": coded_densest,
                    "codes_e": codes_e,
                },
            )
            name = "TRIAL_" if TRIAL_RUN else "entropy"
            wandb.run.name = name + f"Fs {Fs}  Tw {Tw:.2f}"

            # 10 examples stored in the first iteration (visualization)
            ex_idxs = idxs_x_random_per_class(lbls[:code_size])
            ex_img = imgs[ex_idxs]
            ex_cod = codes[ex_idxs]

            # initial log
            log_dict = {
                "will_S": 0.0,
                "ret_AS": coded_AS,
                "ret_%B": coded_densest,
            }
            log_dict["MNIST"] = wandb.Image(np_to_grid(ex_img))
            log_dict["Coded Set"] = wandb.Image(code_grid(ex_cod, K, Q))
            log_dict["Ret Set"] = wandb.Image(code_grid(ex_cod, K, Q))
            wandb.log(log_dict, step=0)

        max_fos = int(codes.shape[0] / codes.shape[1])

        wn = AAWN(code_size)  # empty memory

        for fos in trange(max_fos, desc="storing", unit="fos"):

            n_stored = code_size * (fos + 1)
            new_data = codes[n_stored - code_size : n_stored]

            wn.store(new_data)  # store new data
            ret = wn.retrieve(codes[:n_stored])  # retrieve everything stored so far

            """ measure sparsity """
            ret_AS, ret_densest = measure_sparsity(ret)
            will_S = wn.sparsity()

            """ measure distribution """
            cod_e = shannon_entropy_set(codes[:n_stored].toarray())
            ret_e = shannon_entropy_set(ret[:n_stored].toarray())


            """ Evaluate """
            ret_lbls = lbls[:n_stored]
            err = eval(codes[:n_stored], lbls[:n_stored], ret[:n_stored], ret_lbls)


            if USE_WANDB:
                log_dict = {
                    "will_S": will_S,
                    "ret_AS": ret_AS,
                    "ret_%B": ret_densest,
                    "cod_AS": coded_AS,
                    "cod_%B": coded_densest,
                    "err_pre": err[0],
                    "err_hd_extra": err[1],
                    "err_hd_lost": err[2],
                    "err_hd": err[3],
                    "err_1nn": err[4],
                    "cod_e": cod_e,
                    "ret_e": ret_e,
                }

                ex_ret = ret[ex_idxs]
                log_dict["Ret Set"] = wandb.Image(code_grid(ex_ret, K, Q))

                wandb.log(log_dict, step=n_stored)

        if USE_WANDB:
            wandb.finish()
