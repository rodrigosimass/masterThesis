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
from util.kldiv import *
from tqdm import trange

rng = np.random.RandomState(0)  # reproducible
K = 20
Q = 21
n_epochs = 5
b = 0.8
wta = True

""" Use  Fs = [1,2], Tw = [0.88, 0.91, 0.93] to get results close to the paper."""
list_Fs = [1, 2]
list_Tw = [0.88, 0.91, 0.93]


trial_run = False  # if True: Reduce the size of the datasets for debugging

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

        if trial_run:
            imgs = imgs[: code_size * 2]
            lbls = lbls[: code_size * 2]
            codes = codes[: code_size * 2]

        if USE_WANDB:

            coded_AS, coded_densest = measure_sparsity(codes)

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
                },
            )
            name = "TRIAL_" if trial_run else ""
            wandb.run.name = name + f"Fs {Fs} %B {coded_densest:.4f}"

            # 10 examples stored in the first iteration (visualization)
            ex_idxs = idxs_x_random_per_class(lbls[:code_size])
            ex_img = imgs[ex_idxs]
            ex_cod = codes[ex_idxs]

            # initial log
            log_dict = {
                "will_S": 0,
                "ret_AS": coded_AS,
                "ret_%B": coded_densest,
                "cod_AS": coded_AS,
                "cod_%B": coded_densest,
                "err_1NN": 0.0,
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

            # measure sparsity
            ret_AS, ret_densest = measure_sparsity(ret)
            will_S = wn.sparsity()

            ret_lbls = lbls[:n_stored]

            """ Evaluate """
            err = eval(
                codes[:n_stored], lbls[:n_stored], ret[:n_stored], lbls[:n_stored]
            )

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
                }

                ex_ret = ret[ex_idxs]
                log_dict["Ret Set"] = wandb.Image(code_grid(ex_ret, K, Q))

                wandb.log(log_dict, step=n_stored)
            else:
                print(f"n_stored = {n_stored:.5f}")
                print(f"    pre     = {err[0]:.5f}")
                print(f"    err_1NN_1 = {err[4]:.5f}")
                print(f"    hd      = {err[3]:.5f}")
                print(f"        hd_extra = {err[1]:.5f}")
                print(f"        hd_lost  = {err[2]:.5f}")

        if USE_WANDB:
            wandb.finish()
