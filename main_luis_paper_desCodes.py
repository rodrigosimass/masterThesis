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
from util.whatwhere.description_encoding import *
import time

rng = np.random.RandomState(0)  # reproducible
K = 20
Q = 21
n_epochs = 5
b = 0.8
wta = True

list_Fs = [2]
list_Tw = [0.75, 0.80, 0.85]

"""Noisy-x-hot Description params"""
nxh_x = 500
nxh_Pc = 0.5
nxh_Pr = 0.0

TRIAL_RUN = False  # if True: Reduce the size of the datasets for debugging

imgs, lbls, _, _ = read_mnist(n_train=60000)

if len(sys.argv) < 2:
    print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
    exit(1)
USE_WANDB = bool(int(sys.argv[1]))

descs = noisy_x_hot_encoding(lbls, nxh_x, nxh_Pc, nxh_Pr)
desc_size = descs.shape[1]

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
        coded_AS, coded_densest = measure_sparsity(codes)
        code_size = codes.shape[1]

        """ Attach descriptions to codes """
        desCodes = join(descs, codes)

        if TRIAL_RUN:
            imgs = imgs[: code_size * 2]
            lbls = lbls[: code_size * 2]
            codes = codes[: code_size * 2]
            desCodes = desCodes[: code_size * 2]

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
                    "nxh_x": nxh_x,
                    "nxh_Pc": nxh_Pc,
                    "nxh_Pr": nxh_Pr,
                },
            )
            name = "TRIAL_" if TRIAL_RUN else "desCodes_"
            wandb.run.name = name + f"Fs {Fs}  Tw {Tw:.2f}"

        max_fos = int(codes.shape[0] / codes.shape[1])

        wn_codes = AAWN(code_size)
        wn_desCodes = AAWN(desc_size + code_size)

        for fos in trange(max_fos, desc="storing", unit="fos"):

            n_stored = code_size * (fos + 1)

            new_codes = codes[n_stored - code_size : n_stored]
            new_desCodes = desCodes[n_stored - code_size : n_stored]

            """ Store new data """
            wn_codes.store(new_codes)
            wn_desCodes.store(new_desCodes)

            """ retrieve stored patterns """
            ret_codes = wn_codes.retrieve(codes[:n_stored])

            ret_desCodes = get_codes(
                wn_desCodes.retrieve(desCodes[:n_stored]), desc_size
            )

            """ Measure 1NN classifier error """
            err_1nn_codes = err_1NNclassifier(
                ret_codes, lbls[:n_stored], codes[:n_stored], lbls[:n_stored]
            )
            err_1nn_desCodes = err_1NNclassifier(
                ret_desCodes, lbls[:n_stored], codes[:n_stored], lbls[:n_stored]
            )

            if USE_WANDB:
                log_dict = {
                    "err_1nn_c": err_1nn_codes,
                    "err_1nn_dc": err_1nn_desCodes,
                }

                wandb.log(log_dict, step=n_stored)

        if USE_WANDB:
            wandb.finish()
