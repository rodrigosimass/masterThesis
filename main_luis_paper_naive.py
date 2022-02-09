"""
Reproduction of Sa-Conto and Wichert 2020
"""
import numpy as np
from util.whatwhere.encoder import *
from util.whatwhere.decoder import *
from util.whatwhere.plot import *
from util.whatwhere.naive_baseline import *
from util.willshaw.memory import *
from util.mnist.tools import *
from util.willshaw.plot import *
from util.pickleInterface import *
from util.pytorch.tools import np_to_grid
import wandb
from tqdm import trange


list_B = [3, 4, 6]
list_T = [0.95, 0.9, 0.85]


imgs, lbls, _, _ = read_mnist(n_train=60000)

if len(sys.argv) < 2:
    print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
    exit(1)
USE_WANDB = bool(int(sys.argv[1]))

TRIAL_RUN = False

code_size = 28 * 28

#for B in list_B:
for T in list_T:
    #codes = naive_encode1(imgs, B)
    codes = naive_encode2(imgs, T)

    if TRIAL_RUN:
        imgs = imgs[: code_size * 2]
        lbls = lbls[: code_size * 2]
        codes = codes[: code_size * 2]

    coded_AS, coded_densest = measure_sparsity(codes)

    if USE_WANDB:

        wandb.init(
            project="main_luis_paper",
            entity="rodrigosimass",
            config={"codes_AS": coded_AS, "codes_%B": coded_densest, "naive_T": T},
        )
        name = "TRIAL_" if TRIAL_RUN else ""
        name += "naive2_"
        wandb.run.name = name + f"T {T}  %B {coded_AS:.4f}"

        # initial log
        log_dict = {
            "will_S": 0,
            "ret_AS": coded_AS,
            "ret_%B": coded_densest,
        }
        wandb.log(log_dict, step=0)

    max_fos = 7

    wn = AAWN(code_size)  # empty memory
    for fos in trange(max_fos, desc="storing", unit="fos"):

        n_stored = code_size * (fos + 1)
        new_data = codes[n_stored - code_size : n_stored]

        wn.store(new_data)  # store new data
        ret = wn.retrieve(codes[:n_stored])  # retrieve everything stored so far

        """ measure sparsity """
        ret_AS, ret_densest = measure_sparsity(ret)
        will_S = wn.sparsity()

        """ Evaluate """
        err = eval(codes[:n_stored], lbls[:n_stored], ret[:n_stored], lbls[:n_stored])

        if USE_WANDB:
            log_dict = {
                "will_S": will_S,
                "ret_AS": ret_AS,
                "ret_%B": ret_densest,
                "err_pre": err[0],
                "err_hd_extra": err[1],
                "err_hd_lost": err[2],
                "err_hd": err[3],
                "err_1nn": err[4],
            }

            wandb.log(log_dict, step=(fos + 1))

    if USE_WANDB:
        wandb.finish()
