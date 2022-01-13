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

list_Fs = [2]
list_Tw = [0.95]

"""  params for noisy-x-hot label descriptions """
x = 50  # bits per class
p_c = 0.5  # probability for calss interval
p_r = 0.0  # probability for rest of array


trial_run = False

trn_imgs, trn_lbls, tst_imgs, tst_lbls = read_mnist()

if len(sys.argv) < 2:
    print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
    exit(1)
USE_WANDB = bool(int(sys.argv[1]))

for Tw in list_Tw:
    for Fs in list_Fs:
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

        if trial_run:
            # Reduce the size of the datasets for debugging
            trn_imgs = trn_imgs[: code_size * 2]
            trn_lbls = trn_lbls[: code_size * 2]
            codes = codes[: code_size * 2]
            tst_codes = tst_codes[:1000]
            tst_lbls = tst_lbls[:1000]

        descs = noisy_x_hot_encoding(trn_lbls, x, p_c, p_r)
        tst_descs = noisy_x_hot_encoding(tst_lbls, x, p_c, p_r)

        desc_size = descs.shape[1]
        descs_codes = concatenate(descs, codes)
        tst_descs_codes = concatenate(tst_descs, tst_codes)

        if USE_WANDB:

            coded_AS, coded_densest = measure_sparsity(codes)

            wandb.init(
                project="codes_with_labels",
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
                    "interval_size": x,
                    "p_class": p_c,
                    "p_rest": p_r,
                },
            )
            name = "TRIAL_" if trial_run else ""
            wandb.run.name = name + f"Fs {Fs} %B {coded_densest:.4f}"

            # 10 examples stored in the first iteration (visualization)
            ex_idxs = idxs_x_random_per_class(lbls[:code_size])
            ex_img = trn_imgs[ex_idxs]
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

        wn = AAWN(descs_codes.shape[1])  # memory for codes and descs
        wn_small = AAWN(codes.shape[1])

        x_l = []
        y1_l = []
        y2_l = []
        y3_l = []

        for fos in trange(max_fos, desc="main", unit="fos"):

            n_stored = code_size * (fos + 1)
            new_data = descs_codes[n_stored - code_size : n_stored]
            new_data_small = codes[n_stored - code_size : n_stored]
            
            zero_descs = np.zeros((n_stored, desc_size)) 
            empydesc_codes = concatenate(zero_descs, codes[:n_stored])

            wn.store(new_data)
            wn_small.store(new_data_small)

            ret_normal = wn_small.retrieve(codes[:n_stored])
            
            ret = wn.retrieve(descs_codes[:n_stored])
            ret = deconcatenate(ret, desc_size)[1]
            
            ret_zero = wn.retrieve(empydesc_codes[:n_stored])
            ret_zero = deconcatenate(ret_zero, desc_size)[1]

            err1 = err_1NNclassifier(codes[:n_stored], trn_lbls[:n_stored], ret_normal, trn_lbls[:n_stored])
            err2 = err_1NNclassifier(codes[:n_stored], trn_lbls[:n_stored], ret, trn_lbls[:n_stored])
            err3 = err_1NNclassifier(codes[:n_stored], trn_lbls[:n_stored], ret_zero, trn_lbls[:n_stored])


            x_l.append(n_stored)
            y1_l.append(err1)
            y2_l.append(err2)
            y3_l.append(err3) 
            

            """ aa_r = autoassociation(descs_codes[:n_stored], desc_size, wn)
            compl_r = completion(descs_codes[:n_stored], desc_size, wn)
            class_r = classification(tst_descs_codes, desc_size, wn)

            aa_score = interval_classifier(aa_r, trn_lbls[:n_stored], x)
            compl_score = interval_classifier(compl_r, trn_lbls[:n_stored], x)
            class_score = interval_classifier(class_r, tst_lbls, x)
            
            x_l.append(n_stored)
            y1_l.append(aa_score)
            y2_l.append(compl_score)
            y3_l.append(class_score) """

            if USE_WANDB:
                # step-wise log
                log_dict = {
                    "will_S": will_S,
                    "ret_AS": ret_AS,
                    "ret_%B": ret_densest,
                    "cod_AS": coded_AS,
                    "cod_%B": coded_densest,
                    "err_1NN": err,
                }

                ex_ret = ret[ex_idxs]
                log_dict["Ret Set"] = wandb.Image(code_grid(ex_ret, K, Q))

                wandb.log(log_dict, step=n_stored)


        if USE_WANDB:
            wandb.finish()

    """ label_l = ["Autoassociation", "Completion", "Classification"] """
    label_l = ["normal", "with lbl", "with zero lbl"] 
    
    y_l_l = [y1_l, y2_l, y3_l]
    plot_multiple_line_charts(
        x_l,
        y_l_l,
        label_l,
        xlabel="num_stored",
        ylabel="1NN error",
        #title=f"noisy_x_hot (x={x}, p_c={p_c}, p_r={p_r})",
        path="img/lbl_descs/1NNclassifier4.png",
    )
