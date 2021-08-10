import torch
from numpy.core.numeric import False_
from numpy.lib import utils
from sparse_codes import *
from wn import *
from keras.datasets import mnist
import numpy as np

rng = np.random.RandomState(0)  # reproducible

k = 20  # number of k-means centroids
Fs = 2  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
T_what = 0.8  # Treshod for keeping or discarding a detected feature
Q = 21  # size of the final object space grid
n_epochs = 5  # for the k means feature detection
b = 0.8  # minimum activity of the filters: prevents empty feature detection
wta = True  # winner takes all
factor_of_stored = 2  # stored patterns

(trn_imgs, trn_lbls), _ = mnist.load_data()
trn_imgs = trn_imgs / 255.0

run_name = "k" + str(k) + "_Fs" + str(Fs) + "_ep" + str(n_epochs) + "_b" + str(b)
features = load_or_compute_features(run_name, trn_imgs, k, Fs, rng, n_epochs, b=b)

run_name += "_Q" + str(Q) + "_Tw" + str(T_what)
codes_csr = load_or_compute_codes(
    run_name, trn_imgs, k, Q, features, T_what, wta, trn_lbls
)

will_csr = load_or_compute_will(run_name, codes_csr, factor_of_stored)

ret_csr = load_or_compute_ret(
    trn_imgs,
    features,
    run_name,
    codes_csr,
    will_csr,
    trn_lbls,
    k,
    Q,
    factor_of_stored,
)

print("load success")

num_stored = 784 * factor_of_stored

sim = csr_matrix.dot(ret_csr, codes_csr.T)
nn = csr_matrix.argmax(sim, axis=1)

prediction = trn_lbls[nn]
solution = trn_lbls[: num_stored]

diff = prediction.flatten() - solution.flatten()

errors = np.count_nonzero(diff)

print(
    f"""
Total number of patterns: {diff.shape[0]}
Correct: {diff.shape[0] - errors}
Wrong: {errors}
Error rate: {errors / diff.shape[0]}
"""
)
