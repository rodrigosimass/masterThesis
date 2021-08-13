from numpy.core.numeric import False_
from sparse_codes import *
from wn import *
import numpy as np
from classifier import simple_1NN_classifier
from util import read_mnist

""" in the paper:
k = 20
Q = 21
%B = 0.0068
Fs = 2 radius of receptive fields (size of the receptive fields is 5 * 5)
I = J = 28 (mnist)
"""

rng = np.random.RandomState(0)  # reproducible
""" PARAMETERS """
k = 20  # number of k-means centroids
Fs = 1  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
T_what = 0.8  # Treshod for keeping or discarding a detected feature
Q = 21  # size of the final object space grid
n_epochs = 5  # for the k means feature detection
b = 0.9  # minimum activity of the filters: prevents empty feature detection
wta = True  # winner takes all
factor_of_stored = 3  # stored patterns

small_test = False  # uses only 600 MNIST examples
make_plots = False

trn_imgs, trn_lbls, test_imgs, test_lbls = read_mnist(
    dim=[28, 28], n_train=60000, n_test=1000, one_hot=False
)

run_name = ""

if small_test:
    trn_imgs = trn_imgs[:600, :, :]
    trn_lbls = trn_lbls[:600]
    run_name += "small_"


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