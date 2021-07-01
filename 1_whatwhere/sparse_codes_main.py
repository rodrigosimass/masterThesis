from sparse_codes import *
from Utils import *
from willshaw import *
import sys
import numpy as np

if len(sys.argv) < 2:
    print("USAGE: python3 sparse_codes_main.py verbose")
    sys.exit(0)
verbose = int(sys.argv[1])

""" in the paper:
k = 20
Q = 21
%B = 0.0068
Fs = 2 radius of receptive fields (size of the receptive fields is 5 * 5)
I = J = 28
 """

rng = np.random.RandomState(0)  # reproducible
""" PARAMETERS """
k = 20  # number of k-means centroids
Fs = 2  # size of features, Fs = 1 results in a 3by3 filter size (2Fs+1)
T_what = 0.82  # Treshod for keeping or discarding a detected feature
Q = 21  # size of the final object space grid
n_epochs = 5  # k means
b = 0.8
wta = True
factor_of_stored = 2
""" -----------"""

small_test = False  # uses only 600 MNIST examples

""" Importing train and test data """
trn_imgs = mnist.train_images() / 255.0  # train data
trn_lbls = mnist.train_labels()  # train labels

if small_test:
    trn_imgs = trn_imgs[:600, :, :]
    trn_lbls = trn_lbls[:600]

run_name = (
    "k"
    + str(k)
    + "_Fs"
    + str(Fs)
    + "_ep"
    + str(n_epochs)
    + "_b"
    + str(b)
)
features = load_or_compute_features(
    run_name, trn_imgs, k, Fs, rng, n_epochs, b=b, verbose=verbose
)

run_name += "_Q" + str(Q) + "_Tw" + str(T_what) + "_wta" + str(wta)
codes = load_or_compute_codes(run_name, trn_imgs, k, Q, features, T_what, wta, verbose)

run_name += "_fac" + str(factor_of_stored)
will = load_or_compute_will(run_name, codes, factor_of_stored)

plt.imshow(will, cmap=plt.cm.gray, vmax=1, vmin=0, interpolation="nearest")
plt.show()