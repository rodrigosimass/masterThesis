import numpy as np
import random
import matplotlib.pyplot as plt
import time
import pickle
from scipy.sparse import csr_matrix


def H(vec):
    for i in range(len(vec)):
        if vec[i] < 0:
            vec[i] = 0
        else:
            vec[i] = 1
    return vec


def retreive(cues, W):
    ret = np.empty_like(cues)

    for i in range(cues.shape[0]):  # for all retreival cues
        s = W.dot(cues[i])
        ret[i] = H(s - max(s))

    return ret


def performance(cues, ret):
    hit = 0
    miss = 0
    for i in range(cues.shape[0]):
        if np.array_equal(cues[i], ret[i]):
            hit += 1
        else:
            miss += 1
    print(f"Performance: {hit}/{hit+miss}")


def train(X_trn, M, verbose=0):
    print("X_trn shape: ", X_trn.shape)

    n = X_trn.shape[1]  # size of patterns to memorize
    print("training willshaw of size ", n, "*", n)

    addresses = X_trn[:M]
    print("addresses shape: ", addresses.shape)
    will = np.zeros((n, n))

    for a in addresses:  # for each pattern to store in will
        num_nz = len(a.indices)
        for i in range(num_nz):
            # nz has the indexes of x that are non-zero
            for j in range(i, num_nz):
                idx_i = a.indices[i]
                idx_j = a.indices[j]
                will[idx_i][idx_j] = 1
                will[idx_j][idx_i] = 1

    return will


def load_or_compute_will(run_name, X_trn, factor_of_stored):
    M = 28 * 28 * factor_of_stored
    try:
        will = pickle.load(open(f"sparse_codes_data/{run_name}__will.p", "rb"))
        print(f"loaded will from pickle: sparse_codes_data/{run_name}__will.p")
    except (OSError, IOError) as _:
        will = train(X_trn, M)
        sps_will = csr_matrix(will)
        pickle.dump(sps_will, open(f"sparse_codes_data/{run_name}__will.p", "wb"))
        print(f"saving trained willshaw to sparse_codes_data/{run_name}_will.p")
    return will