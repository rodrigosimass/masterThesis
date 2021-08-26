import numpy as np
from util.whatwhere.encoder import *
from util.willshaw.memory import *
from util.willshaw.classifier import *
from util.mnist.loader import *
import wandb

""" in the paper:
k = 20
Q = 21
%B = 0.0068
Fs = 2 radius of receptive fields (size of the receptive fields is 5 * 5)
I = J = 28 (mnist)
"""

""" Basic PARAMETERS """
k = 20  # number of k-means centroids
Q = 21  # size of the final object space grid
n_epochs = 5  # for the k means feature detection
b = 0.8  # minimum activity of the filters: prevents background feature detection
wta = True  # winner takes all

""" Commonly-adjusted PARAMETERS """
Fs = 2  # square windows of side = 2Fs+1
factor_of_stored = 2  # stored patterns
T_what = 0.95  # Treshod for keeping or discarding a detected feature

wandb.init(project="thesis", entity="rodrigosimass")

""" CONTROLLS """
small_test = False
make_plots = False
verb = True

trn_imgs, trn_lbls, _, _ = read_mnist()

run_name = "test"

if small_test:
    trn_imgs = trn_imgs[:2000, :, :]
    trn_lbls = trn_lbls[:2000]
    run_name += "small_"

rng = np.random.RandomState(0)  # reproducible
run_name += "k" + str(k) + "_Fs" + str(Fs) + "_ep" + str(n_epochs) + "_b" + str(b)
features = load_or_compute_features(
    run_name, trn_imgs, k, Fs, rng, n_epochs, b=b, plot=make_plots, verbose=verb
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
    plot=make_plots,
    verbose=verb,
)

will, will_S = load_or_compute_will(run_name, codes, factor_of_stored, verbose=verb)

ret, ret_AS, ret_densest = load_or_compute_ret(
    trn_imgs,
    features,
    run_name,
    codes,
    will,
    trn_lbls,
    k,
    Q,
    factor_of_stored,
    plot=make_plots,
    verbose=verb,
)

num_stored = 784 * factor_of_stored
class_error = simple_1NN_classifier(ret, codes, trn_lbls, num_stored, verbose=True)