import numpy as np
from util.whatwhere.encoder import *
from util.willshaw.memory import *
from util.willshaw.classifier import *
from util.mnist.loader import *

""" in the paper:
k = 20
Q = 21
%B = 0.0068
Fs = 2 radius of receptive fields (size of the receptive fields is 5 * 5)
I = J = 28 (mnist)
"""

rng = np.random.RandomState(0)  # reproducible

""" Basic PARAMETERS """
k = 20  # number of k-means centroids
Q = 21  # size of the final object space grid
n_epochs = 5  # for the k means feature detection
b = 0.9  # minimum activity of the filters: prevents empty feature detection
wta = True  # winner takes all

""" Commonly-adjusted PARAMETERS """
Fs = 2  # square windows of side = 2Fs+1
factor_of_stored = 2  # stored patterns
T_what = 0.8  # Treshod for keeping or discarding a detected feature

""" CONTROLLS """
small_test = True
make_plots = False

trn_imgs, trn_lbls, _, _ = read_mnist()

run_name = ""

if small_test:
    trn_imgs = trn_imgs[:600, :, :]
    trn_lbls = trn_lbls[:600]
    run_name += "small_"

print(trn_imgs.shape, trn_lbls.shape)


run_name += "k" + str(k) + "_Fs" + str(Fs) + "_ep" + str(n_epochs) + "_b" + str(b)
features = load_or_compute_features(
    run_name, trn_imgs, k, Fs, rng, n_epochs, b=b, plot=make_plots
)

run_name += "_Q" + str(Q) + "_Tw" + str(T_what)
codes_csr, _, _ = load_or_compute_codes(
    run_name, trn_imgs, k, Q, features, T_what, wta, trn_lbls, plot=make_plots
)

will_csr, _ = load_or_compute_will(run_name, codes_csr, factor_of_stored)

ret_csr, _, _ = load_or_compute_ret(
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
err = simple_1NN_classifier(ret_csr, codes_csr, trn_lbls, num_stored, verbose=True)