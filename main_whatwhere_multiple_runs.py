import numpy as np
from util.whatwhere.encoder import *
from util.willshaw.memory import *
from util.willshaw.classifier import *
from util.mnist.loader import *
from util.willshaw.plot import *
import wandb


rng = np.random.RandomState(0)  # reproducible
""" Fixed params """
k = 20  # number of k-means centroids
Q = 21  # size of the final object space grid
n_epochs = 5  # for the k means feature detection
b = 0.8  # minimum activity of the filters: prevents empty feature detection
wta = True  # winner takes all

""" Variable params """
list_Fs = [1, 2]  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
list_Tw = [0.8, 0.9, 0.95]  # Treshod for keeping or discarding a detected feature
step = 5000  # number of patterns stored in willshaw at a time

trn_imgs, trn_lbls, _, _ = read_mnist(n_train=60000)

n_runs = len(list_Fs) * len(list_Tw) * (60000 / step)
idx = 0

for Fs in list_Fs:
    for T_what in list_Tw:

        run_name = (
            "k" + str(k) + "_Fs" + str(Fs) + "_ep" + str(n_epochs) + "_b" + str(b)
        )
        features = load_or_compute_features(
            run_name, trn_imgs, k, Fs, rng, n_epochs, b=b
        )

        run_name += "_Q" + str(Q) + "_Tw" + str(T_what)
        codes, coded_AS, coded_densest = load_or_compute_codes(
            run_name,
            trn_imgs,
            k,
            Q,
            features,
            T_what,
            wta,
            trn_lbls,
        )

        """ wandb.init(
            project="whatwhere",
            entity="rodrigosimass",
            config={
                "km_k": k,
                "km_epochs": n_epochs,
                "km_wta": wta,
                "km_b": b,
                "km_Fs": Fs,
                "ww_Q": Q,
                "ww_Twhat": T_what,
                "codes_AS": coded_AS,
                "codes_%B": coded_densest,
            },
        ) """

        will = None
        ret = None
        for num_stored in range(step, 60000 + 1, step):
            idx += 1
            print(f"{idx} out of {n_runs}")

            new_data = codes[num_stored - step : num_stored]

            will, will_S = incremental_train(new_data, will)

            ret, ret_AS, ret_densest = incremental_retreive(new_data, will, ret)

            class_error = simple_1NN_classifier(
                ret, codes, trn_lbls, num_stored, verbose=True
            )

            err_perfRet = performance_perfect_ret(codes, ret, verbose=True)
            err_infoLoss, err_noise = performance_loss_noise(codes, ret, verbose=True)
            err_avgErr = performance_avg_error(codes, ret, verbose=True)

            """ wandb.log(
                {
                    "ret_AS": ret_AS,
                    "ret_%B": ret_densest,
                    "will_S": will_S,
                    "err_1NN": class_error,
                    "err_perfRet": err_perfRet,
                    "err_infoLoss": err_infoLoss,
                    "err_noise": err_noise,
                    "err_avgErr": err_avgErr,
                },
                step=num_stored,
            ) """

        """ wandb.finish() """