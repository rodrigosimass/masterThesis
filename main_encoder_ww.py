
import numpy as np
from util.willshaw.memory import *
from util.mnist.tools import read_mnist
from util.pickleInterface import *
from util.whatwhere.plot import *
from util.whatwhere.noise import *
from util.whatwhere.description_encoding import *

rng = np.random.RandomState(0)  # reproducible
""" Fixed params """
K = 20  # number of k-means centroids
n_epochs = 5  # for the k means feature detection
b = 0.8  # minimum activity of the filters: prevents empty feature detection
Q = 21  # size of the final object space grid
wta = True  # winner takes all

Fs = 2  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
Tw = 0.9  # Treshod for keeping or discarding a detected feature

cw = False
plot = False

trn_imgs, trn_lbls, tst_imgs, tst_lbls = read_mnist()

features = compute_features(trn_imgs, trn_lbls, K, Fs, rng, n_epochs, b, classwise=cw)

codes, polar = compute_codes(
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

codes_id = get_codes_run_name(K, Fs, n_epochs, b, Q, Tw, wta, cw)
if cw:
    codes_id += "cw"


#%%
if plot:
    plot_features(features, Fs, cw, get_features_run_name(K, Fs, n_epochs, b, cw))
    plot_feature_maps(codes, trn_lbls, K, Q, codes_id)
    plot_recon_examples(
        trn_imgs, trn_lbls, codes, K, Q, codes_id + "new", features, polar
    )
    plot_mnist_codes_activity(trn_imgs, codes, K, Q, codes_id)
    codes_salt = add_zero_noise(codes, prob=0.2)
    codes_pepper = add_one_noise(codes, prob=0.01)
    plot_recon_examples(
        trn_imgs, trn_lbls, codes_salt, K, Q, "salt_new", features, polar
    )
    plot_recon_examples(
        trn_imgs, trn_lbls, codes_pepper, K, Q, "pepper_new", features, polar
    )
