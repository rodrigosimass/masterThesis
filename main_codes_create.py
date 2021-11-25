import numpy as np
from util.willshaw.memory import *
from util.mnist.tools import read_mnist
from util.pickleInterface import *
from util.whatwhere.plot import *
from util.whatwhere.noise import *

rng = np.random.RandomState(0)  # reproducible
""" Fixed params """
K = 21  # number of k-means centroids
n_epochs = 5  # for the k means feature detection
b = 0.8  # minimum activity of the filters: prevents empty feature detection
Q = 21  # size of the final object space grid
wta = True  # winner takes all

Fs = 4  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
T_what = 0.8  # Treshod for keeping or discarding a detected feature
cw = False


trn_imgs, trn_lbls, tst_imgs, _ = read_mnist()

features = compute_features(
    trn_imgs, trn_lbls, K, Fs, rng, n_epochs, b, verbose=True, classwise=cw
)
norms = np.linalg.norm(features, axis=(1,2))

plot_features(features, Fs, cw, get_features_run_name(K, Fs, n_epochs, b, cw))

codes, polar = compute_codes(
    trn_imgs,
    K,
    Q,
    features,
    T_what,
    wta,
    n_epochs,
    b,
    Fs,
    verbose=True,
    set="trn",
)

codes_id = get_codes_run_name(K, Fs, n_epochs, b, Q, T_what, wta, cw)
if cw:
    codes_id += "cw"

"""
plot_mnist_codes_activity(trn_imgs, codes, K, Q, codes_id)
plot_feature_maps(codes, trn_lbls, K, Q, codes_id) """

plot_recon_examples(trn_imgs, trn_lbls, codes, K, Q, codes_id + "new", features, polar)

codes_salt = add_zero_noise(codes, prob=0.2)
codes_pepper = add_one_noise(codes, prob=0.01)
plot_recon_examples(trn_imgs, trn_lbls, codes_salt, K, Q, "salt_new", features, polar)
plot_recon_examples(
    trn_imgs, trn_lbls, codes_pepper, K, Q, "pepper_new", features, polar
)
