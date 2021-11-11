import numpy as np
from util.whatwhere.encoder import *
from util.willshaw.memory import *
from util.mnist.tools import read_mnist
from util.pickleInterface import *

rng = np.random.RandomState(0)  # reproducible
""" Fixed params """
K = 40  # number of k-means centroids
Fs = 3  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
n_epochs = 5  # for the k means feature detection
b = 0.8  # minimum activity of the filters: prevents empty feature detection
Q = 12  # size of the final object space grid
wta = True  # winner takes all
T_what = 0.9  # Treshod for keeping or discarding a detected feature
cw = True


trn_imgs, trn_lbls, tst_imgs, _ = read_mnist()

features = compute_features(
    trn_imgs, trn_lbls, K, Fs, rng, n_epochs, b, verbose=True, classwise=cw
)
print(features.shape)
plot_features(features, Fs, get_features_run_name(K, Fs, n_epochs, b, cw))

codes, _ = compute_codes(
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

codes_id = get_codes_run_name(K, Fs, n_epochs, b, Q, T_what)

# whole dataset
plot_class_activity_1D_stacked(codes, trn_lbls, K, Q, codes_id)
plot_mnist_codes_activity(trn_imgs, codes, K, Q, codes_id)
plot_class_activity_2D(codes, trn_lbls, K, Q, codes_id)
# plot_class_activity_1D(codes, trn_lbls, K, Q, codes_id)
# plot_sparsity_distribution(codes, K, Q, codes_id)

# one data point
# plot_feature_maps_overlaped(trn_imgs, codes, K, Q, codes_id)
# plot_feature_maps(codes, K, Q, codes_id)
