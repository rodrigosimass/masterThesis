import numpy as np
from util.whatwhere.encoder import *
from util.whatwhere.decoder import *
from util.whatwhere.plot import *
from util.willshaw.memory import *
from util.mnist.tools import *
from util.willshaw.plot import *
from util.pickleInterface import *
from util.pytorch.tools import np_to_grid
from util.whatwhere.description_encoding import *
import wandb
from util.kldiv import *
from tqdm import trange

""" params for codes """
rng = np.random.RandomState(0)  # reproducible
K = 20
Q = 21
n_epochs = 5
b = 0.8
wta = True

Fs = 2
Tw = 0.75

"""  params for noisy-x-hot label descriptions """
l_nxh_x = [400, 650, 850]  # bits per class
l_nxh_probs = [[0.5, 0.0]]  # pairs (Pc,Pr)


TRIAL_RUN = False

trn_imgs, trn_lbls, tst_imgs, tst_lbls = read_mnist()

if len(sys.argv) < 2:
    print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
    exit(1)
USE_WANDB = bool(int(sys.argv[1]))


features = compute_features(trn_imgs, trn_lbls, K, Fs, rng, n_epochs, b)

codes, _ = compute_codes(
    trn_imgs,
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

tst_codes, _ = compute_codes(
    tst_imgs,
    K,
    Q,
    features,
    Tw,
    wta,
    n_epochs,
    b,
    Fs,
    set="tst",
)

code_size = codes.shape[1]

if TRIAL_RUN:
    # Reduce the size of the datasets for debugging
    trn_imgs = trn_imgs[: code_size * 2]
    trn_lbls = trn_lbls[: code_size * 2]
    codes = codes[: code_size * 2]
    tst_codes = tst_codes[:1000]
    tst_lbls = tst_lbls[:1000]

""" Grid Search over description parameters """
for nxh_x in l_nxh_x:
    for nxh_Pc, nxh_Pr in l_nxh_probs:

        """Create Descriptions"""
        descs = noisy_x_hot_encoding(trn_lbls, nxh_x, nxh_Pc, nxh_Pr)
        tst_descs = noisy_x_hot_encoding(tst_lbls, nxh_x, nxh_Pc, nxh_Pr)
        desc_size = descs.shape[1]

        """ Concatenate description with codes"""
        desCodes = join(descs, codes)
        tst_desCodes = join(tst_descs, tst_codes)

        coded_AS, coded_densest = measure_sparsity(codes)

        if USE_WANDB:
            wandb.init(
                project="classifier_WN",
                entity="rodrigosimass",
                config={
                    "km_K": K,
                    "km_epochs": n_epochs,
                    "km_wta": wta,
                    "km_b": b,
                    "km_Fs": Fs,
                    "ww_Q": Q,
                    "ww_Tw": Tw,
                    "codes_AS": coded_AS,
                    "codes_%B": coded_densest,
                    "nxh_x": nxh_x,
                    "nxh_Pc": nxh_Pc,
                    "nxh_Pr": nxh_Pr,
                },
            )
            name = "TRIAL_" if TRIAL_RUN else ""
            name += "nxhGS_"
            wandb.run.name = name + f"x {nxh_x}  Pc {nxh_Pc}  Pr {nxh_Pr}"

        max_fos = int(codes.shape[0] / codes.shape[1])

        wn = AAWN(desCodes.shape[1])

        for fos in trange(max_fos, desc="storing", unit="fos"):
            n_stored = code_size * (fos + 1)
            new_data = desCodes[n_stored - code_size : n_stored]

            wn.store(new_data)

            """ Delete the description from de desCodes """
            empty_desCodes = delete_descs(desCodes[:n_stored], desc_size)
            empty_tst_desCodes = delete_descs(tst_desCodes, desc_size)

            """ Retrieve """
            ret_descs_aa = get_descs(
                wn.retrieve(desCodes[:n_stored]), desc_size
            )  # AutoAssociation
            ret_descs_trn = get_descs(
                wn.retrieve(empty_desCodes[:n_stored]), desc_size
            )  # Classification (trn)
            ret_descs_tst = get_descs(
                wn.retrieve(empty_tst_desCodes[:n_stored]), desc_size
            )  # Classification (tst)

            """ Evaluate """
            aa_acc = interval_classifier(ret_descs_aa, trn_lbls[:n_stored], nxh_x)
            trn_acc = interval_classifier(ret_descs_trn, trn_lbls[:n_stored], nxh_x)
            tst_acc = interval_classifier(ret_descs_tst, tst_lbls, nxh_x)

            if USE_WANDB:
                log_dict = {
                    "aa_acc": aa_acc,
                    "trn_acc": trn_acc,
                    "tst_acc": tst_acc,
                }

                wandb.log(log_dict, step=fos + 1)

        if USE_WANDB:
            wandb.finish()
