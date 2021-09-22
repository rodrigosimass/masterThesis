import numpy as np
from util.whatwhere.encoder import *
from util.willshaw.memory import *
from util.willshaw.classifier import *
from util.mnist.loader import *
from util.willshaw.plot import *
from util.pickleInterface import *
import wandb


rng = np.random.RandomState(0)  # reproducible
""" Fixed params """
K = 20  # number of k-means centroids
Q = 21  # size of the final object space grid
n_epochs = 5  # for the k means feature detection
b = 0.8  # minimum activity of the filters: prevents empty feature detection
wta = True  # winner takes all

""" Variable params """
Fs = 2  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
Tw = 0.95


trn_imgs, trn_lbls, tst_imgs, tst_lbls = read_mnist(n_train=60000, n_test=10000)


features = load_or_compute_features(trn_imgs, K, Fs, rng, n_epochs, b)

ww_trn, ww_tst, _, _ = load_or_compute_codes(
    trn_imgs, tst_imgs, K, Q, features, Tw, wta, n_epochs, b, Fs, test=True
)

ret = load_ret(k, Q, num_stored, Fs, n_epochs, b, Tw)