import numpy as np
from util.whatwhere.encoder import *
from util.willshaw.memory import *
from util.willshaw.classifier import *
from util.mnist.loader import *
from util.willshaw.plot import *


def run(
    run_name,
    trn_imgs,
    k,
    Fs,
    rng,
    n_epochs,
    b,
    Q,
    T_what,
    wta,
    trn_lbls,
    factor_of_stored,
    make_plots,
):
    run_name += "k" + str(k) + "_Fs" + str(Fs) + "_ep" + str(n_epochs) + "_b" + str(b)
    features = load_or_compute_features(
        run_name, trn_imgs, k, Fs, rng, n_epochs, b=b, plot=make_plots
    )

    run_name += "_Q" + str(Q) + "_Tw" + str(T_what)
    codes_csr, coded_AS, coded_densest = load_or_compute_codes(
        run_name, trn_imgs, k, Q, features, T_what, wta, trn_lbls, plot=make_plots
    )

    will_csr, will_S = load_or_compute_will(run_name, codes_csr, factor_of_stored)

    ret_csr, ret_AS, ret_densest = load_or_compute_ret(
        trn_imgs,
        features,
        run_name,
        codes_csr,
        will_csr,
        trn_lbls,
        k,
        Q,
        factor_of_stored,
        plot=make_plots,
    )

    num_stored = 784 * factor_of_stored
    class_error = simple_1NN_classifier(
        ret_csr, codes_csr, trn_lbls, num_stored, verbose=True
    )

    return (coded_AS, coded_densest, ret_AS, ret_densest, will_S, class_error)


rng = np.random.RandomState(0)  # reproducible
""" PARAMETERS """
k = 20  # number of k-means centroids
Q = 21  # size of the final object space grid
n_epochs = 5  # for the k means feature detection
b = 0.9  # minimum activity of the filters: prevents empty feature detection
wta = True  # winner takes all

make_plots = False
small_test = False

if small_test:
    trn_imgs, trn_lbls, _, _ = read_mnist(n_train=6000)
else:
    trn_imgs, trn_lbls, _, _ = read_mnist(n_train=60000)

list_Fs = [1, 2, 3]  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
list_Tw = [
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
]  # Treshod for keeping or discarding a detected feature
list_fos = [1, 3, 5, 6, 7]  # stored patterns

f = open("results/multiple_runs.csv", "w")
f.write(
    "k,n_epochs,b,Q,wta,Fs,T_what,factor_of_stored,Coded_AS,Coded_%B,Ret_AS,Ret_%B,Will_S,Class_error\n"
)

l_will_S = []
l_l_class_error = []
l_labels = []

for Fs in list_Fs:
    for T_what in list_Tw:
        l_class_error = []
        for factor_of_stored in list_fos:
            run_name = ""
            line = f"{k},{n_epochs},{b},{Q},{wta},{Fs},{T_what},{factor_of_stored},"
            if small_test:
                run_name += "small_"
            coded_AS, coded_densest, ret_AS, ret_densest, will_S, class_error = run(
                run_name,
                trn_imgs,
                k,
                Fs,
                rng,
                n_epochs,
                b,
                Q,
                T_what,
                wta,
                trn_lbls,
                factor_of_stored,
                make_plots,
            )

            l_class_error.append(class_error)

            line += f"{round(coded_AS,4)},{round(coded_densest,4)},{round(ret_AS,4)},{round(ret_densest,4)},{round(will_S,4)},{round(class_error,4)}\n"
            f.write(line)
        l_l_class_error.append(l_class_error)
        l_labels.append(f"Fs={Fs},%B={round(coded_AS,4)}")

""" plot_multiple_line_charts(
    list_fos,
    l_l_class_error,
    l_labels,
    "classification_error",
    "factor of stored",
    "classification error",
) """
