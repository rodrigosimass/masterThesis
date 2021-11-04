import numpy as np
from util.whatwhere.encoder import *
from util.willshaw.memory import *
from util.mnist.tools import read_mnist
from util.pickleInterface import *

rng = np.random.RandomState(0)  # reproducible
""" Fixed params """
K = 30  # number of k-means centroids
Q = 18  # size of the final object space grid
n_epochs = 5  # for the k means feature detection
b = 0.8  # minimum activity of the filters: prevents empty feature detection
wta = True  # winner takes all
Fs = 2  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
T_what = 0.6  # Treshod for keeping or discarding a detected feature


trn_imgs, trn_lbls, tst_imgs, tst_lbls = read_mnist()

features = compute_features(trn_imgs, K, Fs, rng, n_epochs, b, verbose=True)
plot_features(features, Fs, get_features_run_name(K, Fs, n_epochs, b))

codes, _, AS, densest = compute_codes(
    trn_imgs,
    tst_imgs,
    K,
    Q,
    features,
    T_what,
    wta,
    n_epochs,
    b,
    Fs,
    verbose=True,
    test=True,
)

codes_id = get_codes_run_name(K, Fs, n_epochs, b, Q, T_what)

plot_class_activity_2D(codes, trn_lbls, K, Q, codes_id)
plot_feature_maps_overlaped(trn_imgs, codes, K, Q, codes_id)
plot_feature_maps(codes, K, Q, codes_id)
plot_sparsity_distribution(codes, K, Q, codes_id, AS, densest)
plot_mnist_codes_activity(trn_imgs, codes, K, Q, codes_id)
